import tensorflow as tf
import os
import math
import numpy as np 
import hyperparams as hyp
# from tensorflow.python.framework import ops
# from flow_transformer import transformer
# from PIL import Image
# from scipy.misc import imsave
# from math import pi
# from skimage.draw import *
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

EPS = 1e-6

def stop_execution(t, msg = ''):
    def f(t):
        print msg
        exit()
        return t
    return tf.py_func(f, [t], t.dtype)

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def normalize(d):
    dmin = tf.reduce_min(d)
    dmax = tf.reduce_max(d)
    d = (d-dmin)/(dmax-dmin)
    return d

def normalize_within_ex(d):
    return tf.map_fn(normalize, (d), dtype=tf.float32)

def oned2color(d,norm=True):
    # convert a 1chan input to a 3chan image output
    # (it's not very colorful yet)
    if norm:
        d = normalize(d)
    return tf.cast(tf.tile(255*d,[1,1,1,3]),tf.uint8)

def oned2red(d,norm=True):
    # convert a 1chan input to a 3chan image output
    # (it's not very colorful yet)
    if norm:
        d = normalize(d)
    red = tf.cast(255*d,tf.uint8)
    # put ones at the "red" pixels, so that "> 0" works for every channel
    zero = red/128
    rgb = tf.concat([red, zero, zero],axis=3)
    return rgb

def preprocess_color(x):
    return tf.cast(x,tf.float32) * 1./255 - 0.5

def back2color(i):
    return tf.cast((i+0.5)*255,tf.uint8)

def back2gray(i):
    r, g, b = tf.split(i, 3, axis=3)
    rgb = tf.tile((r+g+b)/3,[1,1,1,3])
    return back2color(rgb)

def match(xs, ys): #sort of like a nested zip
    result = {}
    for i, (x,y) in enumerate(zip(xs, ys)):
        if type(x) == type([]):
            subresult = match(x, y)
            result.update(subresult)
        else:
            result[x] = y
    return result
    
def feed_from(inputs, variables, sess):
    return match(variables, sess.run(inputs))

def add_loss(loss_dict, loss, name):
    tf.summary.scalar('%s' % name, loss)
    loss_dict.update({'%s' % name: loss})
    return loss_dict
    
def batch_confusion(cat, pred_cat):
    print_shape(cat)
    print_shape(pred_cat)
    cat = tf.expand_dims(cat, axis=1)
    pred_cat = tf.expand_dims(pred_cat, axis=1)
    print_shape(cat)
    print_shape(pred_cat)
    return tf.map_fn(single_confusion, (cat, pred_cat), dtype=tf.int32)

def single_confusion((cat, pred_cat)):
    print_shape(cat)
    print_shape(pred_cat)
    conf = tf.confusion_matrix(cat, pred_cat, num_classes=hyp.nCats)
    conf = tf.squeeze(conf)
    return conf
    
def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
