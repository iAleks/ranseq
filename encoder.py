import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
import hyperparams as hyp

def encoder(inputs, pred_dim, name, nLayers, relu=False, std=1e-4, do_decode=True, is_train=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        shape = inputs.get_shape()
        B = int(shape[0])
        # H = int(shape[1])
        # W = int(shape[2])
        
        # with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
        #                     padding="VALID",
        #                     activation_fn=tf.nn.relu,
        #                     normalizer_fn=slim.batch_norm,
        #                     normalizer_params={'is_training':is_train,
        #                                        'decay':0.97, # 997
        #                                        'epsilon':1e-5,
        #                                        'scale':True,
        #                                        'updates_collections':None},
        #                     stride=1,                                     
        #                     weights_initializer=tf.truncated_normal_initializer(stddev=std),
        #                     weights_regularizer=slim.l2_regularizer(0.05)):
        #     # ENCODER
        #     net = inputs
        #     chans = 32
        #     # first, one conv at full res
        #     net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
        #     net = slim.conv2d(net, chans, [3, 3], stride=1,
        #                       scope='conv%d' % 0)
        #     print_shape(net)
        #     for i in range(nLayers):
        #         chans = int(chans*2)
        #         net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
        #         net = slim.conv2d(net, chans, [3, 3], stride=2,
        #                           scope='conv%d_1' % (i+1))
        #         print_shape(net)
        #         net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
        #         net = slim.conv2d(net, chans, [3, 3], stride=1,
        #                           scope='conv%d_2' % (i+1))
        #         print_shape(net)
        #         H = int(H/2)
        #         W = int(W/2)
        # # add a fully-connected layer
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.05)):
            # if is_train and hyp.do_dropout:
            #     net = slim.dropout(net, 0.5)
            # net = tf.nn.relu(slim.fully_connected(tf.reshape(net, [hyp.B,-1]), 100))
            net = inputs
            print_shape(net)
            if is_train and hyp.do_dropout_input:
                net = slim.dropout(net, 0.5)
            net = slim.fully_connected(tf.reshape(net, [hyp.B,-1]), 100)
            print_shape(net)
            if is_train and hyp.do_dropout:
                net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, pred_dim)
            print_shape(net)
        
            # net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyp.pad)
            # net = slim.conv2d(net, pred_dim, [3, 3], stride=1, scope='pred')
            # print_shape(net)
            # pred = tf.reduce_mean(net, axis=[1,2])
            # print_shape(pred)
    return net
