import tensorflow as tf
from tqdm import tqdm

from Client import Clients

def buildClients(num):
    learning_rate = 0.0001
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    num_classes = 10  # Cifar-10 total classes (0-9 digits)

    #create Client and model
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num)


def run_global_test(client, global_vars, test_num):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))


#### SOME TRAINING PARAMS ####
CLIENT_NUMBER = 100
CLIENT_RATIO_PER_ROUND = 0.12
epoch = 360


#### CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER)

#### BEGIN TRAINING ####
global_vars = client.get_client_vars()
for ep in range(epoch):
    # We are going to sum up active clients' vars at each epoch
    client_vars_sum = None

    # Choose some clients that will train on this epoch
    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)

    # Train with these clients
    for client_id in tqdm(random_clients, ascii=True):
        # Restore global vars to client's model
        client.set_global_vars(global_vars)

        # train one client
        client.train_epoch(cid=client_id)

        # obtain current client's vars
        current_client_vars = client.get_client_vars()

        # sum it up
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += ccv

    # obtain the avg vars as global vars
    global_vars = []
    for var in client_vars_sum:
        global_vars.append(var / len(random_clients))

    # run test on 600 instances
    run_global_test(client, global_vars, test_num=600)


#### FINAL TEST ####
run_global_test(client, global_vars, test_num=10000)