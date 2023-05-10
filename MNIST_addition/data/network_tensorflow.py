import tensorflow as tf
from tensorflow.keras import layers

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from pathlib import Path

import numpy as np

class Net(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(Net, self).__init__()

        # encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(6, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(layers.Conv2D(16, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(layers.Flatten())
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(120))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(10))

    def call(self, inputs, training=True):
        x = self.encoder(inputs)
        if training:
            x = self.dropout_layer(x)
        x = self.classifier(x)
        return x
    
class Net_NTP(tf.keras.Model):
    def __init__(self, dataset, dropout_rate):
        super(Net_NTP, self).__init__()

        # encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(6, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(layers.Conv2D(16, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(layers.Flatten())
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(120))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(100))
        self.classifier.add(tf.keras.layers.Activation("relu"))

        self.DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        if dataset == "mnist":
            self.dataset = {
                "train": torchvision.datasets.MNIST(
                    root=str(self.DATA_ROOT), train=True, download=True, transform=self.transform
                ),
                "test": torchvision.datasets.MNIST(
                    root=str(self.DATA_ROOT), train=False, download=True, transform=self.transform
                ),
            }
        elif dataset == "fashion_mnist":
            self.dataset = {
                "train": torchvision.datasets.FashionMNIST(
                    root=str(self.DATA_ROOT), train=True, download=True, transform=self.transform
                ),
                "test": torchvision.datasets.FashionMNIST(
                    root=str(self.DATA_ROOT), train=False, download=True, transform=self.transform
                ),
            }

        self.x_values_train, self.x_values_test = get_mnist_data_as_numpy(self.dataset)

    def call(self, inputs, training=True):
        x = inputs
        if x[0] == 'd':
            labels = tf.constant([int(x[1])])
            one_hot = tf.one_hot(labels, 100)
            # float64_tensor = tf.cast(one_hot, dtype=tf.float32)
            return one_hot

        else:
            # x is an image
            split = x.split('-')[0]
            ind = int(x.split('-')[1])
            # image, _ = self.dataset[split][ind]

            if split =="train":
                image = tf.convert_to_tensor(self.x_values_train[ind])
                
            elif split =="test":
                image = tf.convert_to_tensor(self.x_values_test[ind])
            expanded_tensor = tf.expand_dims(image, axis=0)
            x = self.encoder(expanded_tensor)
            if training:
                x = self.dropout_layer(x)
            x = self.classifier(x)
            # float64_tensor = tf.cast(x, dtype=tf.float32)
            return x
        
def get_mnist_data_as_numpy(datasets):
    img_train = datasets["train"].data.numpy()
    img_train = img_train[:, :, :, np.newaxis]
    img_train = img_train/255.0
    img_test = datasets["test"].data.numpy()
    img_test = img_test/255.0
    img_test = img_test[:, :, :, np.newaxis]
    
    return img_train, img_test