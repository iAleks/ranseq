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
G = 20499
n = 160
N = n*n
# nGenes = 10

root_dir = 'profile_images'
mkdir(root_dir)

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
# for ind in range(nUniques):
#     name = names[ind]
#     # out_dir = '%s/%s' % (root_dir, name)

for part in range(nParts):
    print 'reading part %d/%d' % (part, nParts-1)
    # covars_file = 'covars_all/part%02d' % part
    # with open(covars_file,'r') as f:
    #     reader = csv.reader(f,delimiter='\t')
    #     genes = list(reader)
    #     nGenes = len(genes)
    #     print 'found %d genes' % nGenes
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
    # raw_input()

        # if label==name:
        #     # if count < maxPngs
        #     gene = genes[ind]
        #     gene = np.asarray(gene).astype(np.float32)
        #     # print 'gene %d; type %s' % (ind, label)
        #     count += 1
    # print 'found %d gene profiles with the name %s' % (count, name)
        
# for ind in range(1,nGenes): # start at 1 to skip header
#     gene = genes[ind]
#     gene = np.asarray(gene).astype(np.float32)
#     label = labels[ind][0]
#     print label
#     print '-'*100
#     print 'gene %d; type %s' % (ind, label)
    
#     gene = gene + 1
#     gene = np.log(gene)
#     print np.amax(gene)
#     print np.mean(gene)
#     print np.median(gene)
#     print np.amin(gene)

#     gene = np.pad(gene, (0, N-G), 'constant')
#     gene = np.reshape(gene, [n, n])

    
#     imsave('%04d.png' % ind, gene)
#     # hist, bin_edges = np.histogram(gene, density=True)
#     # print hist
#     # print bin_edges

#     # plt.hist(gene, bins=20)
#     # plt.title('hist')
#     # plt.show()
#     # raw_input()
