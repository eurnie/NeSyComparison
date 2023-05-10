import tensorflow as tf
from tensorflow.keras import layers

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