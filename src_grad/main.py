from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

def split_data(path, clients_num):
    # 读取数据
    data = pd.read_csv(path)
    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,
        data["Occupancy"].values.reshape(-1, 1),
        random_state=42)
    
    # one-hot 编码
    y_train = np.concatenate([1 - y_train, y_train], 1)
    y_test = np.concatenate([1 - y_test, y_test], 1)
    
    # 训练集划分给多个client
    X_train = np.array_split(X_train, clients_num)
    y_train = np.array_split(y_train, clients_num)
    return X_train, X_test, y_train, y_test

CLIENT_NUM = 6
X_train, X_test, y_train, y_test = split_data("./data/datatraining.txt", CLIENT_NUM)


import os
import pickle
import gzip

BASE_DIR = "./storage"

if not os.path.isdir(BASE_DIR):
    os.mkdir(BASE_DIR)

def pack(model):
    pkl = pickle.dumps(model)
    pkl = gzip.compress(pkl)
    return pkl


def unpack(data):
    pkl = gzip.decompress(data)
    model = pickle.loads(pkl)
    return model


def client_query_model():
    """return the newest model and epoch num"""
    
    newest_epoch = -1
    res_f = None
    
    for f in os.listdir(BASE_DIR):
        if not f.startswith('global_model'):
            continue
        file_name = os.path.splitext(f)[0]
        epoch = int(file_name.split('_')[-1])
        
        if epoch > newest_epoch:
            newest_epoch = epoch
            res_f = f
    
    # file found
    with open("{}/{}".format(BASE_DIR, res_f), 'rb') as rf:
        res = rf.read()
    
    return unpack(res), newest_epoch


def client_upload_one_update(update, epoch, c_id):
    """upload one model update"""
    
    file_name = "{}/local_update_{}_{}.ieen".format(BASE_DIR, c_id, epoch)
    data = pack(update)
    
    with open(file_name, 'wb') as wf:
        wf.write(data)
    
    return


def server_query_updates(cur_epoch):
    """query all model updates"""
    
    res = []
    
    for f in os.listdir(BASE_DIR):
        if not f.startswith('local_update'):
            continue
        file_name = os.path.splitext(f)[0]
        epoch = int(file_name.split('_')[-1])
        
        if epoch == cur_epoch:
            with open("{}/{}".format(BASE_DIR, f), 'rb') as rf:
                data = unpack(rf.read())
                res.append(data)
    
    return res


def server_upload_model(model, epoch):
    """upload one model with epoch num"""
    
    file_name = "{}/global_model_{}.ieen".format(BASE_DIR, epoch)
    data = pack(model)
    
    with open(file_name, 'wb') as wf:
        wf.write(data)
        
    return


# client 要训练的epoch
client_epoch = [0] * CLIENT_NUM
client_learning_rate = 0.001

def train_model(client_id):
    model, epoch = client_query_model()
    if epoch < client_epoch[client_id]:
        return
    
    tf.compat.v1.reset_default_graph()
    
    n_samples = X_train[client_id].shape[0]
    
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])
    
    ser_W, ser_b = model
    W = tf.Variable(ser_W)
    b = tf.Variable(ser_b)

    pred = tf.matmul(x, W) + b

    # 定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,
    															labels=y))

    # 梯度下降
#     optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(client_learning_rate)
    
    gradient = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(gradient)

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 训练模型
    with tf.Session() as sess:
        sess.run(init)
        
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            _, c = sess.run(
                [train_op, cost],
                feed_dict={
                    x: X_train[client_id][i * batch_size:(i + 1) * batch_size],
                    y: y_train[client_id][i * batch_size:(i + 1) * batch_size, :]
                })
            avg_cost += c / total_batch
    
        # 获取更新量
        val_W, val_b = sess.run([W, b])
    
    delta_W = (ser_W-val_W)/client_learning_rate
    delta_b = (ser_b-val_b)/client_learning_rate
    delta_model = [delta_W, delta_b]
    meta = [n_samples, avg_cost]
    
    client_upload_one_update([delta_model, meta], epoch, client_id)
    
    client_epoch[client_id] = epoch
    return



# 跑测试集
def testing(ser_W, ser_b):
    tf.compat.v1.reset_default_graph()
    
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])
    
    W = tf.Variable(ser_W)
    b = tf.Variable(ser_b)
    pred = tf.matmul(x, W) + b
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 跑模型
    with tf.Session() as sess:
        sess.run(init)
        acc = accuracy.eval({x: X_test, y: y_test})
    
    return acc

# 设置模型
batch_size = 100
n_features = 5
n_class = 2

EPOCH_NUM = 50 * CLIENT_NUM
server_lr = 0.001

# 模型参数
server_W = np.zeros([n_features, n_class], dtype=np.float32)
server_b = np.zeros([n_class], dtype=np.float32)
server_model = [server_W, server_b]

for epoch in range(EPOCH_NUM):
    server_upload_model(server_model, epoch)
    
    for c_id in range(CLIENT_NUM):
        train_model(c_id)
    
    total_grad_W = None
    total_grad_b = None
    total_size = 0
    total_cost = 0
    
    updates = server_query_updates(epoch)
    for update in updates:
        grads, meta = update
        grad_W, grad_b = grads
        data_size, cost = meta
        
        total_grad_W = (grad_W * data_size) if (total_grad_W is None) else (total_grad_W + grad_W * data_size)
        total_grad_b = (grad_b * data_size) if (total_grad_b is None) else (total_grad_b + grad_b * data_size)
        total_size += data_size
        total_cost += cost
        
    total_grad_W /= total_size
    total_grad_b /= total_size
    total_cost /= CLIENT_NUM
    
    
    # update global model
    server_W = server_W - server_lr * total_grad_W
    server_b = server_b - server_lr * total_grad_b
    server_model = [server_W, server_b]
    
    test_acc = testing(server_W, server_b)
    print("Epoch: {:03}, cost: {:.2f}, test_acc: {:.4f}".format(epoch, total_cost, test_acc))