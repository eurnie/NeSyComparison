from tensorflow.keras import layers
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
class Net_NTP(tf.keras.Model):
    def __init__(self, vocab, dataset, dropout_rate):
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
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(10))

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

        self.vocab = vocab

        # original embeddings
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initializer = tf.compat.v1.random_uniform_initializer(-1.0, 1.0)

        self.embedding_matrix_predicates = \
            tf.compat.v1.get_variable(
                "embeddings", [len(vocab), 10],
                initializer=initializer
            )

        # print('--------------------------------')
        # print(tf.nn.embedding_lookup(params=self.embedding_matrix_predicates, ids=[1, 2, 3]))
        # print(self.call([1, 2, 3]))
        # print('--------------------------------')

    def call(self, inputs, training=True):
        output = tf.Variable(tf.zeros((0, 10), dtype=tf.float32))
        for i, ind in enumerate(inputs):
            sym = self.vocab.get_sym(ind)

            if sym in ['digit', 'unknown', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 
                    'sixteen', 'seventeen', 'eighteen', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
                    'd7', 'd8', 'd9']:
                row = tf.reshape(self.embedding_matrix_predicates[ind], (1, 10))
            elif sym is None:
                emb = tf.Variable(tf.zeros((1, 10), dtype=tf.float32))
                row = tf.reshape(emb, (1, 10))

            else:
                split = sym.split('-')[0]
                image_i = int(sym.split('-')[1])

                if split =="train":
                    image = tf.convert_to_tensor(self.x_values_train[image_i])
                    
                elif split =="test":
                    image = tf.convert_to_tensor(self.x_values_test[image_i])
                expanded_tensor = tf.expand_dims(image, axis=0)
                x = self.encoder(expanded_tensor)
                if training:
                    x = self.dropout_layer(x)
                row = tf.reshape(self.classifier(x), (1, 10))
                row = tf.cast(row, dtype=tf.float32)

            output = tf.concat([output, row], axis=0)

        return output
    
def get_mnist_data_as_numpy(datasets):
    img_train = datasets["train"].data.numpy()
    img_train = img_train[:, :, :, np.newaxis]
    img_train = img_train/255.0
    img_test = datasets["test"].data.numpy()
    img_test = img_test/255.0
    img_test = img_test[:, :, :, np.newaxis]
    
    return img_train, img_test