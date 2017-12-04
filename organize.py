import csv
import numpy as np
import os.path
from scipy.misc import imsave
import matplotlib.pyplot as plt

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def shorten(name):
    return ''.join(e for e in name if e.isalnum())


unique_file = 'labels_unique.txt';

with open(unique_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    uniques = list(reader)
    nCats = len(uniques)
    print 'found %d unique labels' % nCats
    cats = []
    for ind in range(nCats):
        uniq = shorten(uniques[ind][0])
        print uniques[ind]
        print uniq
        cats.append(uniq)

print cats        
nParts = 43 # hardcode this for now
cat_counts = np.zeros(nCats)

for part in range(nParts):
    print 'reading part %d/%d' % (part, nParts-1)
    labels_file = 'labels_all/part%02d' % part
    with open(labels_file,'r') as f:
        reader = csv.reader(f,delimiter='\n')
        labels = list(reader)
        nLabels = len(labels)
        print 'found %d labels' % nLabels
    if part==0:
        labels = labels[1:]
        nLabels = nLabels - 1
    for label_ind in range(nLabels):
        label = shorten(labels[label_ind][0])
        cat_ind = cats.index(label)
        print 'part %02d/%d; label %d/%d; label %s; cat index %d' % (part, nParts-1,
                                                                    label_ind, nLabels-1,
                                                                    label,
                                                                    cat_ind)
        cat_counts[cat_ind] += 1
        print cat_counts
    print '-'*100
data_dir = 'data'
mkdir(data_dir)    
np.save('%s/cat_counts.npy' % data_dir, cat_counts)
np.save('%s/cats.npy' % data_dir, cats)
