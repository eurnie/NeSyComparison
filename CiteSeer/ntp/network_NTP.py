import torch_geometric
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from pathlib import Path

class Net_NTP(tf.keras.Model):
    def __init__(self, vocab, dataset, dropout_rate):
        super(Net_NTP, self).__init__()

        # dropout layer
        self.dropout_layer = layers.Dropout(rate=dropout_rate)

        # classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(layers.Dense(840))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(84))
        self.classifier.add(tf.keras.layers.Activation("relu"))
        self.classifier.add(layers.Dense(20))

        # import dataset
        DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
        self.data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
        self.citation_graph = self.data[0]

        # vocabulary that matches all the predicates and entities to ids
        self.vocab = vocab

        # initialize embeddings
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        initializer = tf.compat.v1.random_uniform_initializer(-1.0, 1.0)
        self.embedding_matrix = \
            tf.compat.v1.get_variable(
                "embeddings", [len(vocab), 20],
                initializer=initializer
            )

        self.embedding_matrix = self.update_embedding_matrix()

    # update the embeddings by giving all the documents to the neural network
    def update_embedding_matrix(self):

        id = []
        id.append(self.vocab.get_id('doc_100'))

        print(f'This is the initial embedding of doc_100 (with index {id[0]}):')
        symbol_embedded = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=id)
        tf.print(symbol_embedded)

        for i in range(len(self.vocab)):
            sym = self.vocab.get_sym(i)
            flag = True
            try:
                int(sym)
            except ValueError:
                # sym is not an integer
                flag = False

            if (not flag) and (sym not in ['document_label', 'cite', 'unknown', '<UNK>']):
                document_index = int(sym.split('_')[1])
                document_features = self.citation_graph.x[document_index]
                expanded_tensor = tf.expand_dims(document_features, axis=0)
                # if training:
                #     x = self.dropout_layer(x)
                x = self.classifier(expanded_tensor)
                a = self.embedding_matrix
                self.embedding_matrix = tf.concat(axis=0, values=[a[:i], x, a[i+1:]])

        print(f'This is the new embedding of doc_100 (with index {id[0]}):')
        symbol_embedded = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=id)
        tf.print(symbol_embedded)

        print('-- embedding matrix updated --')
        return self.embedding_matrix