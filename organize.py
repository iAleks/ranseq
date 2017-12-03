nimport csv
import numpy as np
import os.path
from scipy.misc import imsave
import matplotlib.pyplot as plt

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def shorten(name):
    return ''.join(e for e in name if e.isalnum())

names=[]
ages=[]

unique_file = 'labels_unique.txt';
covars_file = 'covars_all/part00';
labels_file = 'labels_all/part00';
G = 20499
n = 160
N = n*n
# nGenes = 10

root_dir = 'profile_images'
mkdir(root_dir)

with open(covars_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    genes = list(reader)
    nGenes = len(genes)
    print 'found %d genes' % nGenes
with open(labels_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    labels = list(reader)
    nLabels = len(labels)
    print 'found %d labels' % nLabels
assert(nGenes==nLabels)
with open(unique_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    uniques = list(reader)
    nUniques = len(uniques)
    print 'found %d unique labels' % nUniques
    names = []
    for ind in range(1,nUniques): # start at 1 to skip header
        uniq = shorten(uniques[ind][0])
        names.append(uniq)

maxPngs = 100
for ind in range(nUniques):
    name = names[ind]
    # out_dir = '%s/%s' % (root_dir, name)
    
    # find all profiles with this name
    for ind in range(1,nGenes): # start at 1 to skip header
        label = shorten(labels[ind][0])
        count = 0
        if label==name:
            # if count < maxPngs
            gene = genes[ind]
            gene = np.asarray(gene).astype(np.float32)
            # print 'gene %d; type %s' % (ind, label)
            count += 1
    print 'found %d gene profiles with the name %s' % (count, name)
    raw_input()
        
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
