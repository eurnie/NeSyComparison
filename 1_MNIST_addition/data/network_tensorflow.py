import tensorflow as tf
from tensorflow.keras import layers

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(6, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("elu"))
        self.encoder.add(layers.Conv2D(16, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("elu"))
        self.encoder.add(layers.Flatten())
        
        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(120))
        self.classifier.add(tf.keras.layers.Activation("elu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("elu"))
        self.classifier.add(layers.Dense(10))
        # self.classifier.add(tf.keras.layers.Activation('softmax'))

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.classifier(x)
        return x

class Net_Dropout(tf.keras.Model):
    def __init__(self):
        super(Net_Dropout, self).__init__()

        # encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(6, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("elu"))
        self.encoder.add(layers.Conv2D(16, 5))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(tf.keras.layers.Activation("elu"))
        self.encoder.add(layers.Flatten())
        
        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(120))
        self.classifier.add(tf.keras.layers.Activation("elu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("elu"))
        self.classifier.add(layers.Dense(10))
        # self.classifier.add(tf.keras.layers.Activation('softmax'))

    def call(self, inputs, training=False):

        x = self.encoder(inputs)

        if training:
            dropout_layer = layers.Dropout(0.99)
            x = dropout_layer(x)

        import sys
        tf.print(x, output_stream=sys.stderr)
        x = self.classifier(x)
        return x

class Net_Original(tf.keras.Model):
    def __init__(self):
        super(Net_Original, self).__init__()

        # encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Conv2D(6,5,activation="elu"))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(layers.Conv2D(16,5,activation="elu"))
        self.encoder.add(layers.MaxPool2D((2,2)))
        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.Dense(100, activation="elu"))

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(84, activation="elu"))
        self.classifier.add(layers.Dense(10))

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.classifier(x)
        return x