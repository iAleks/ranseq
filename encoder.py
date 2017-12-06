import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
import hyperparams as hyp

def encoder(inputs, pred_dim, name, is_train=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        shape = inputs.get_shape()
        if hyp.do_batch_balance:
            B = hyp.nCats*hyp.B
        else:
            B = hyp.B

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.005)):
            inputs = tf.reshape(inputs, [B, -1])
            if is_train and hyp.do_dropout_input:
                inputs = slim.dropout(inputs, 0.5)
            net = tf.nn.relu(slim.fully_connected(inputs, 100))
            if is_train and hyp.do_dropout:
                net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, pred_dim)
    return net
