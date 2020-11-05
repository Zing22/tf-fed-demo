# tf-fed-demo
`./src`: A federated learning demo for AlexNet on CIFAR-10 dataset, basing on Tensorflow.\\
`./src_grad`: A simple TF demo for federated LR with uploading the gradients from clients to server.

## Dependence
1. Python 3.7
2. Tensorflow v1.14.x
3. tqdm
4. jupyter

## Usage

Upload updated models version:
```bash
cd ./src
python Server.py
```

Upload model update gradients version:

```bash
cd ./src_grad
python main.py
```
or
```bash
cd ./src_grad
jupyter notebook
```

## Blog
* Upload updated models: [https://blog.csdn.net/Mr_Zing/article/details/101938334](https://blog.csdn.net/Mr_Zing/article/details/101938334)
* Upload model update gradients: [https://blog.csdn.net/Mr_Zing/article/details/109496824](https://blog.csdn.net/Mr_Zing/article/details/109496824)