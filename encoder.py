import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
import hyperparams as hyp

def encoder(inputs, pred_dim, name, nLayers, relu=False, std=1e-4, do_decode=True, is_train=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        shape = inputs.get_shape()
        if hyp.do_batch_balance:
            B = hyp.nCats*hyp.B
        else:
            B = hyp.B
        #                     activation_fn=tf.nn.relu,
        with slim.arg_scope([slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.05)):
            # net = tf.nn.relu(slim.fully_connected(tf.reshape(net, [B,-1]), 100))
            net = inputs
            print_shape(net)
            if is_train and hyp.do_dropout_input:
                net = slim.dropout(net, 0.5)
            net = tf.reshape(net, [B,-1])
            print_shape(net)
            net = tf.nn.relu(slim.fully_connected(net, 100))
            print_shape(net)
            if is_train and hyp.do_dropout:
                net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, pred_dim)
            print_shape(net)

    return net
