import tensorflow as tf
# from utils import *
import hyperparams as hyp
import os
# from augs import *
from utils import *
from readers import *

# sys.path.append('../writers_readers')
# from read_svkitti_tfrecords import *

def batch(dataset,B,catid=0,shuffle=True):
    print dataset
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print 'found %d records' % nRecords
    if nRecords == 0:
        gene = tf.zeros([hyp.N, 1], tf.float32)
        # cat = catid*tf.ones([], tf.int64)
        cat = -1*tf.ones([], tf.int64)
    else:        
        for record in records:
            assert os.path.isfile(record), 'Record at %s was not found' % record

        queue = tf.train.string_input_producer(records, shuffle=shuffle)

        (gene,cat) = read_and_decode(queue)

        # gene = tf.cast(gene,tf.float32)
        gene = tf.reshape(gene,[hyp.N, 1])
        if hyp.do_log:
            gene = tf.log(1+gene)
        if not hyp.mult_noise_std==0:
            mult = tf.random_normal([hyp.N,1],1,hyp.mult_noise_std)
            gene = gene*mult
        if not hyp.add_noise_std==0:
            noise = tf.random_normal([hyp.N,1],0,hyp.add_noise_std)
            gene = gene + noise
        if hyp.do_normalize:
            gene = normalize(gene)-0.5
    batch = tf.train.batch([gene,cat],batch_size=B)
    return batch
