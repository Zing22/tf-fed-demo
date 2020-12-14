import numpy as np
from tensorflow.keras.utils import to_categorical


class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        perm = self.random_order[self.start:self.start + batch_size]

        self.start += batch_size
        if self.start > self.size:
            self.start = 0

        return self.x[perm], self.y[perm]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, load_data_func, one_hot=True, split=0):
        (x_train, y_train), (x_test, y_test) = load_data_func()
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        if one_hot:
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
        else:
            self.train = self.splited_batch(x_train, y_train, split)

        self.test = BatchGenerator(x_test, y_test)

    def splited_batch(self, x_data, y_data, split):
        res = []
        for x, y in zip(np.split(x_data, split), np.split(y_data, split)):
            assert len(x) == len(y)
            res.append(BatchGenerator(x, y))
        return res
