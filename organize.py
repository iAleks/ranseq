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
    for ind in range(1,nCats): # start at 1 to skip header
        uniq = shorten(uniques[ind][0])
        cats.append(uniq)

print cats        
maxPngs = 100
nParts = 43 # hardcode this for now

cat_counts = np.zeros(nCats)
print cat_counts

for part in range(nParts):
    print 'reading part %d/%d' % (part, nParts-1)
    labels_file = 'labels_all/part%02d' % part
    with open(labels_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        labels = list(reader)
        nLabels = len(labels)
        print 'found %d labels' % nLabels
    # assert(nGenes==nLabels)
    nGenes = nLabels

    if part==0:
        start=1 # start at 1 to skip header
    else:
        start=0
    for gene_ind in range(start,nGenes):
        label = shorten(labels[gene_ind][0])
        cat_ind = cats.index(label)
        print 'part %02d/%d; gene %d/%d; label %s; cat index %d' % (part, nParts-1,
                                                                    gene_ind, nGenes-1,
                                                                    label,
                                                                    cat_ind)
        cat_counts[cat_ind] += 1
        print cat_counts
    print '-'*100
data_dir = 'data'
mkdir(data_dir)    
np.save('%s/cat_counts.npy' % data_dir, cat_counts)
np.save('%s/cats.npy' % data_dir, cats)
