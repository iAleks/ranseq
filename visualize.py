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

data_dir = 'data'
cat_counts = np.load('%s/cat_counts.npy' % data_dir)
cats = np.load('%s/cats.npy' % data_dir).tolist()
print cat_counts
print cats
nCats = len(cat_counts)

G = 20499
n = 160
N = n*n
# nGenes = 10

root_dir = 'profile_images'
mkdir(root_dir)
maxPngs = 100
nParts = 43 # hardcode this for now

# nProfiles = 5
print cats.index('CL0000057fibroblast')

png_counts = np.zeros(nCats)
for part in range(nParts):
    print 'reading part %d/%d' % (part, nParts-1)
    covars_file = 'covars_all/part%02d' % part
    with open(covars_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        genes = list(reader)
        nGenes = len(genes)
        print 'found %d genes' % nGenes
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
        cat = shorten(labels[gene_ind][0])
        cat_ind = cats.index(cat)
        print 'part %02d/%d; gene %d/%d; label %s; cat index %d' % (part, nParts-1,
                                                                    gene_ind, nGenes-1,
                                                                    cat,
                                                                    cat_ind)
        if cat_counts[cat_ind] > 100 and png_counts[cat_ind] < 100:
            print 'found example %d of cat %d, %s' % (png_counts[cat_ind], cat_ind, cat)
            gene = genes[gene_ind]
            gene = np.asarray(gene).astype(np.float32)
            gene = np.log(gene+1)
            gene = np.pad(gene, (0, N-G), 'constant')
            gene = np.reshape(gene, [n, n])
            out_dir = '%s/%s' % (root_dir, cat)
            mkdir(out_dir)
            imsave('%s/%04d.png' % (out_dir, png_counts[cat_ind]), gene)
            png_counts[cat_ind] += 1
    print '-'*100
