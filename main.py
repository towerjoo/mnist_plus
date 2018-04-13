import os
from keras import layers
import numpy as np
from PIL import Image


def load_data():
    train_dir = "datasets/train/"
    test_dir = "datasets/train/"
    train_data = []
    train_labels = []
    for f in os.listdir(train_dir):
        fh = Image.open(os.path.join(train_dir, f))
        fh.load()
        sample = np.asarray(fh, dtype="int32")
        train_data.append(sample)
        train_labels.append(f.split("-")[0])
    test_data = []
    test_labels = []
    for f in os.listdir(test_dir):
        fh = Image.open(os.path.join(train_dir, f))
        fh.load()
        sample = np.asarray(fh, dtype="int32")
        test_data.append(sample)
        test_labels.append(f.split("-")[0])
    return (train_data, train_labels), (test_data, test_labels)
        
def gen_model():
    # we can use ConvNet to do the magic
    pass

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_data()
