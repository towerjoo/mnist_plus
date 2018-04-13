import random
import os
import uuid
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image


def gen_data(M=5000, N=1000):
    # N: the number of images to generate
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_dir = "datasets/train/"
    test_dir = "datasets/test/"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir) # create if not exists
    if not os.path.exists(test_dir):
        os.makedirs(test_dir) # create if not exists
        
    for i in range(M+N):
        index = random.choice(range(x_train.shape[0]))
        im = Image.fromarray(x_train[index])
        name = str(y_train[index])

        # apply resizing
        resize_ratio = random.random() * .5 + .5 # .5 <= r <= 1
        newsize = (int(im.size[0] * resize_ratio), int(im.size[1] * resize_ratio)) # PIL requires size to be int
        im = im.resize(newsize)

        # apply rotation
        rotate_degree = random.randint(0, 91) # 0<=d<=90
        im = im.rotate(rotate_degree)
        im = im.resize((28, 28)) # to keep the original size 
        if i < M:
            path = "{}{}-{}.jpg".format(train_dir, name, uuid.uuid4().hex)
        else:
            path = "{}{}-{}.jpg".format(test_dir, name, uuid.uuid4().hex)
        im.save(path)


if __name__ == "__main__":
    gen_data()
