import tensorflow as tf

class CustomEmbedding:
    def __init__(self, vocab, emb_pred, emb_const, predicate_ids, constant_ids):
        self.vocab = vocab
        self.emb_pred = emb_pred
        self.emb_const = emb_const
        self.predicate_ids = predicate_ids
        self.constant_ids = constant_ids

    def __call__(self, index):
        return_list = []

        for i in index:
            if i in self.predicate_ids:
                i_embedded = tf.nn.embedding_lookup(params=self.emb_pred, ids=[i])      
            else:
                original_ind = self.vocab.get_sym(i)
                i_embedded = self.emb_const(original_ind)
            return_list.append(i_embedded)

        return return_list