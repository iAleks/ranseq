import tensorflow as tf
import hyperparams as hyp

def get_placeholders(model):
    if hyp.do_batch_balance:
        model.gene = tf.placeholder(tf.float32,[hyp.B*hyp.nCats,hyp.N,1], name='gene')
        model.cat = tf.placeholder(tf.int64,[hyp.B*hyp.nCats], name='cat')
    else:
        model.gene = tf.placeholder(tf.float32,[hyp.B,hyp.N,1], name='gene')
        model.cat = tf.placeholder(tf.int64,[hyp.B], name='cat')

    model.placeholders = [
        model.gene,
        model.cat,
    ]
    return model.placeholders
