import tensorflow as tf
from tensorflow.keras import layers

class Net_CiteSeer(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(Net_CiteSeer, self).__init__()
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

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
    
class Net_Cora(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(Net_Cora, self).__init__()
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(840))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(7))

    def call(self, inputs, training=True):
        if training:
            inputs = self.dropout_layer(inputs)
        return self.classifier(inputs)
    
class Net_PubMed(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(Net_PubMed, self).__init__()
        
        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(840))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(3))

    def call(self, inputs, training=True):
        if training:
            inputs = self.dropout_layer(inputs)
        return self.classifier(inputs)