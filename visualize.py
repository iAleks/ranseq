import csv
import numpy as np
import os.path
from scipy.misc import imsave

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
n = 160
N = n*n
# nGenes = 10

root_dir = 'profile_images'
mkdir(root_dir)
minPngs = 0
maxPngs = 50
nParts = 43 # hardcode this for now

# png_counts = np.zeros([nEids,nCats])
count = 0
cat_map = -1*np.ones(nCats)
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
        print 'part %02d/%d; gene %d/%d; cat %s; cat index %d; eid %s; nCats %d' % (part, nParts-1,
                                                                                    gene_ind, nGenes-1,
                                                                                    cat,
                                                                                    cat_ind,
                                                                                    eid,
                                                                                    np.amax(cat_map)+1)
        # if cat_counts[cat_ind] > minPngs and png_counts[int(eid),cat_ind] < maxPngs:
        if cat_map[cat_ind]==-1:
            cat_map[cat_ind] = cat_iterator
            print 'adding category %d' % cat_iterator
            cat_iterator += 1
        cat_id = cat_map[cat_ind]
        # print 'found example %d of cat %d, %s' % (png_counts[int(eid), cat_ind], cat_ind, cat)
        gene = genes[gene_ind]
        gene = np.asarray(gene).astype(np.float32)
        gene = np.log(gene+1)
        gene = np.pad(gene, (0, N-G), 'constant')
        gene = np.reshape(gene, [n, n])
        out_dir = '%s/%s/%04d' % (root_dir, eid, cat_id)
        mkdir(out_dir)
        imsave('%s/%06d.png' % (out_dir, count), gene)
        # png_counts[int(eid), cat_ind] += 1
        count += 1
    print '-'*100
print 'total %d categories' % (np.amax(cat_map)+1)
