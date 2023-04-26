import tensorflow as tf
from tensorflow.keras import layers

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
       
        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(840))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(6))

    def call(self, inputs, training=True):
        return self.classifier(inputs)
        

class Net_Dropout(tf.keras.Model):
    def __init__(self):
        super(Net_Dropout, self).__init__()
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=0.2)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(840))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(6))

    def call(self, inputs, training=True):
        if training:
            inputs = self.dropout_layer(inputs)
        return self.classifier(inputs)