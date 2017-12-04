import tensorflow as tf
import hyperparams as hyp
import batcher

def get_inputs(model):
    model.gene_t = tf.ones((hyp.B, hyp.N, 1))
    model.gene_v = tf.ones((hyp.B, hyp.N, 1))
    model.cat_t = tf.ones((hyp.B), tf.int64)
    model.cat_v = tf.ones((hyp.B), tf.int64)

    (model.gene_t,
     model.cat_t) = batcher.batch(hyp.dataset_t,
                                  hyp.B,
                                  shuffle=True)
    (model.gene_v,
     model.cat_v) = batcher.batch(hyp.dataset_v,
                                  hyp.B,
                                  shuffle=True)
    model.train_inputs = [
        model.gene_t,
        model.cat_t,
    ]

    model.val_inputs = [
        model.gene_v,
        model.cat_v,
    ]
    
    return model.train_inputs, model.val_inputs

