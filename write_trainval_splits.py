import csv
import numpy as np
import os.path
import random
import glob
from scipy.misc import imsave

image_dir = 'profile_images'
tfr_dir = 'data/tfrs'
data_dir = 'data'
cat_counts = np.load('%s/cat_counts.npy' % data_dir)
cats = np.load('%s/cats.npy' % data_dir).tolist()
print cat_counts
print cats
nCats = 46

e_ids = os.listdir(image_dir)
nExperiments = len(e_ids)

nVersions = 5
max_per_cat = 100000
for v in range(nVersions):
    cat_examples = [[] for _ in range(nCats)]
    train_eids = np.load('%s/e_ids_train_%d.npy' % (data_dir, v))
    print train_eids
    for eid in train_eids:
        pattern = '%s/%s*.tfrecord' % (tfr_dir, eid)
        # print pattern
        examples = sorted(glob.glob(pattern))
        for ex in examples:
            pieces = ex.split('_')
            cat = int(pieces[1])
            cat_examples[cat].append(ex)
    f0 = open("%s/train_%d.txt" % (data_dir, v), "w")
    for c in range(nCats):
        f = open("%s/train_%d_%d.txt" % (data_dir, v, c), "w")
        examples = cat_examples[c]
        random.shuffle(examples)
        for e in range(np.amin([max_per_cat, len(examples)])):
            s = examples[e]
            s = s[len(tfr_dir)+1:]
            f.write('%s\n' % s)
            f0.write('%s\n' % s)
        f.close()
    f0.close()
    
    val_eids = np.load('%s/e_ids_val_%d.npy' % (data_dir, v))
    print val_eids
    cat_examples = [[] for _ in range(nCats)]
    for eid in val_eids:
        pattern = '%s/%s*.tfrecord' % (tfr_dir, eid)
        examples = sorted(glob.glob(pattern))
        for ex in examples:
            pieces = ex.split('_')
            cat = int(pieces[1])
            cat_examples[cat].append(ex)
    f0 = open("%s/val_%d.txt" % (data_dir, v), "w")
    for c in range(nCats):
        f = open("%s/val_%d_%d.txt" % (data_dir, v, c), "w")
        examples = cat_examples[c]
        random.shuffle(examples)
        for e in range(np.amin([max_per_cat, len(examples)])):
            s = examples[e]
            s = s[len(tfr_dir)+1:]
            f.write('%s\n' % s)
            f0.write('%s\n' % s)
        f.close()
    f0.close()
    print 'done version %d' % v
    print '-'*100
    # raw_input()
        
        
