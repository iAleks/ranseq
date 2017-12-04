import csv
from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import os.path

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def shorten(name):
    return ''.join(e for e in name if e.isalnum())

data_dir = 'data'
cat_counts = np.load('%s/cat_counts.npy' % data_dir)
cats = np.load('%s/cats.npy' % data_dir).tolist()
print cat_counts
print cats
nCats = len(cat_counts)

with open("eids_unique.txt",'r') as f:
    reader = csv.reader(f,delimiter='\n')
    eids = list(reader)
eids = [int(eid[0]) for eid in eids]
nEids = len(eids)
print nEids
print eids

G = 20499

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

out_dir = "data/tfrs2"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

nParts = 43 # hardcode this for now
im_id = 0
cat_map = -1*np.ones(nCats, np.int64)
cat_iterator = 0
for part in range(nParts):
    print 'reading part %d/%d' % (part, nParts-1)
    covars_file = 'covars_all/part%02d' % part
    with open(covars_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        genes = list(reader)
        nGenes = len(genes)
        # print 'found %d genes' % nGenes
    labels_file = 'labels_all/part%02d' % part
    with open(labels_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        labels = list(reader)
        nLabels = len(labels)
        # print 'found %d labels' % nLabels
    eids_file = 'ids_all/part%02d' % part
    with open(eids_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        eids = list(reader)
        nEids = len(eids)
        # print 'found %d eids' % nEids
    # assert(nGenes==nLabels)
    nGenes = nLabels
    if part==0:
        start=1 # start at 1 to skip header
    else:
        start=0
    for gene_ind in range(start,nGenes):
        cat = shorten(labels[gene_ind][0])
        cat_ind = cats.index(cat)
        eid = shorten(eids[gene_ind][0])
        if cat_map[cat_ind]==-1:
            cat_map[cat_ind] = cat_iterator
            print 'adding category %d' % cat_iterator
            cat_iterator += 1
        cat_id = cat_map[cat_ind]
        
        out_file = "%s/%s_%04d_%08d.tfrecord" % (out_dir,eid,cat_id,im_id)
        im_id += 1
        if os.path.isfile(out_file):
            sys.stdout.write(':')
        else:
            gene = genes[gene_ind]
            gene = np.asarray(gene).astype(np.float32)
            gene_raw = gene.tostring()
            compress = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP)
            writer = tf.python_io.TFRecordWriter(out_file, options=compress)
            example = tf.train.Example(features=tf.train.Features(feature={
                'gene_raw': _bytes_feature(gene_raw),
                'cat_id': _int64_feature(cat_id),
            }))
            writer.write(example.SerializeToString())
            writer.close()
            sys.stdout.write('.')
        sys.stdout.flush()
print 'done'      
