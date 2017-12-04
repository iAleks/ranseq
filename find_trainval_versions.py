import csv
import numpy as np
import os.path
import random
from scipy.misc import imsave

image_dir = 'profile_images'
data_dir = 'data'
cat_counts = np.load('%s/cat_counts.npy' % data_dir)
cats = np.load('%s/cats.npy' % data_dir).tolist()
print cat_counts
print cats
nCats = 46

e_ids = os.listdir(image_dir)
nExperiments = len(e_ids)

version = 0
print 'searching for version %d' % version
while version < 10:
    random.shuffle(e_ids)
    e_ids_train = e_ids[:int(np.floor(nExperiments*0.8))]
    e_ids_val = e_ids[int(np.floor(nExperiments*0.8)):]
    e_ids_train.sort()
    e_ids_val.sort()
    # print e_ids_train
    # print e_ids_val
    cat_coverage = np.zeros(nCats, np.int64)
    for e_id in e_ids_train:
        e_id_dir = '%s/%s' % (image_dir, e_id)
        c_ids = os.listdir(e_id_dir)
        for c_id in c_ids:
            cat_dir = '%s/%s' % (e_id_dir, c_id)
            cat_id = int(c_id)
            # print 'looking at %s; cat_id = %d' % (cat_dir, cat_id)
            if os.path.isdir(cat_dir):
                ims = os.listdir(cat_dir)
                nIms = len(ims)
                cat_coverage[cat_id] += nIms
    # print cat_coverage
    cat_cover = np.sum(cat_coverage > 0)
    # print 'this covers %d/%d cats' % (cat_cover, nCats)
    if cat_cover==nCats:
        print 'got a valid version!'
        print e_ids_train
        exists = 0
        for v in range(version):
            otherv = np.load('%s/e_ids_train_%d.npy' % (data_dir, v))
            eq = otherv==e_ids_train
            if np.sum(eq)==len(eq):
                exists = 1
                print 'this seems to be an exact copy of version %d' % v
        if not exists:
            print 'this version does not yet exist! saving!'
            np.save('%s/e_ids_train_%d.npy' % (data_dir, version), e_ids_train)
            np.save('%s/e_ids_val_%d.npy' % (data_dir, version), e_ids_val)
            version += 1
        print 'searching for version %d' % version
        # raw_input()

        
