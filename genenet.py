from scipy.misc import imsave
import cv2
import tensorflow as tf
import hyperparams as hyp
from encoder import *
from utils import *
import sys
from math import pi

EPS = 1e-6

def GeneNet(gene, cat, is_train=True, reuse=False):
    loss_dict = {}
    shape = gene.get_shape()
    B = int(shape[0])
    N = hyp.N
    with tf.variable_scope("gene"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        pred = encoder(gene, hyp.nCats, "GeneNet",
                       is_train=is_train,
                       reuse=reuse)
        print_shape(pred)
        print_shape(cat)
        
        inds = tf.where(cat > -1)
        cat = tf.squeeze(tf.gather(cat, inds), axis=1)
        pred = tf.squeeze(tf.gather(pred, inds), axis=1)

        label = tf.one_hot(cat,hyp.nCats,axis=1)
        # print_shape(label)
        # cat = tf.Print(cat, [cat], 'cat', summarize=100)
        # pred = tf.Print(pred, [pred], 'pred', summarize=100)
        # label = tf.Print(label, [label], 'label', summarize=100)

        ce = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
        # ce = tf.Print(ce, [ce], 'ce', summarize=100)
        ce = tf.reduce_mean(ce)
        # print_shape(ce)
        # ce = tf.reduce_mean(ce)
        # print_shape(ce)
        loss_dict = add_loss(loss_dict, ce, 'ce_loss')

        # pred is B x hyp.nCats
        pred_cat = tf.cast(tf.argmax(pred,axis=1), tf.int64)
        # pred_class is B
        # print_shape(pred_cat)
        correct = tf.equal(pred_cat, cat)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        cm = batch_confusion(cat, pred_cat)
        cm = tf.reduce_mean(tf.cast(cm, tf.float32), axis=0)
        cm = tf.reshape(cm, [1, hyp.nCats, hyp.nCats, 1])
        cm = oned2color(cm)
        tf.summary.image('confusion_matrix', cm)

    return loss_dict, pred_cat, cat
