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
while version < 5:
    random.shuffle(e_ids)
    proportion = 0.9
    e_ids_train = e_ids[:int(np.floor(nExperiments*proportion))]
    e_ids_val = e_ids[int(np.floor(nExperiments*proportion)):]
    e_ids_train.sort()
    e_ids_val.sort()
    # print e_ids_train
    # print e_ids_val
    cat_coverage_t = np.zeros(nCats, np.int64)
    cat_coverage_v = np.zeros(nCats, np.int64)
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
                cat_coverage_t[cat_id] += nIms
    for e_id in e_ids_val:
        e_id_dir = '%s/%s' % (image_dir, e_id)
        c_ids = os.listdir(e_id_dir)
        for c_id in c_ids:
            cat_dir = '%s/%s' % (e_id_dir, c_id)
            cat_id = int(c_id)
            # print 'looking at %s; cat_id = %d' % (cat_dir, cat_id)
            if os.path.isdir(cat_dir):
                ims = os.listdir(cat_dir)
                nIms = len(ims)
                cat_coverage_v[cat_id] += nIms
    cat_cover_t = np.sum(cat_coverage_t > 0)
    cat_cover_v = np.sum(cat_coverage_v > 0)
    cat_cover_both = np.sum((cat_coverage_v>0) == (cat_coverage_t>0))
    # print ((cat_coverage_t>0).astype(int) - (cat_coverage_v>0).astype(int))
    # if np.float32(cat_cover_t)/nCats > 0.68 and np.float32(cat_cover_v)/nCats > 0.68:
    if cat_cover_t==nCats and np.float32(cat_cover_both)/nCats > 0.2:
        print 'got a valid version!'
        print 'this covers %d/%d cats in train (%.2f), and %d/%d cats in val (%.2f), and %d/%d cats in both (%.2f)' % (
            cat_cover_t, nCats, np.float32(cat_cover_t)/nCats,
            cat_cover_v, nCats, np.float32(cat_cover_v)/nCats,
            cat_cover_both, nCats, np.float32(cat_cover_both)/nCats)
        print ((cat_coverage_t>0).astype(int) - (cat_coverage_v>0).astype(int))
        print e_ids_train
        print e_ids_val
        exists = 0
        for v in range(version):
            otherv = np.load('%s/e_ids_train_%d.npy' % (data_dir, v))
            eq = otherv==e_ids_train
            if np.sum(eq)==len(eq):
                exists = 1
                print '!'*100
                print 'this seems to be an exact copy of version %d' % v
        if not exists:
            print 'this version does not yet exist! saving!'
            np.save('%s/cat_cover_both_%d.npy' % (data_dir, version), cat_cover_both)
            np.save('%s/e_ids_train_%d.npy' % (data_dir, version), e_ids_train)
            np.save('%s/e_ids_val_%d.npy' % (data_dir, version), e_ids_val)
            version += 1
        print 'searching for version %d' % version
        # raw_input()

        
