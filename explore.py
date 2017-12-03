import csv
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

names=[]
ages=[]

covars_file = 'covars_100.txt';
labels_file = 'labels_100.txt';

G = 20499
n = 160
N = n*n
# nGenes = 10

with open(covars_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    genes = list(reader)
    nGenes = len(genes)
    print nGenes
with open(labels_file,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    labels = list(reader)
    nLabels = len(labels)
    print nLabels
assert(nGenes==nLabels)

print 'we have %d genes' % nGenes
for ind in range(1,nGenes): # start at 1 to skip header
    gene = genes[ind]
    gene = np.asarray(gene).astype(np.float32)
    label = labels[ind][0]
    print label
    print '-'*100
    print 'gene %d; type %s' % (ind, label)
    print np.amax(gene)
    print np.mean(gene)
    print np.median(gene)
    print np.amin(gene)
    
    gene = gene + 1
    gene = np.log(gene)
    print np.amax(gene)
    print np.mean(gene)
    print np.median(gene)
    print np.amin(gene)

    gene = np.pad(gene, (0, N-G), 'constant')
    gene = np.reshape(gene, [n, n])
    
    imsave('%04d.png' % ind, gene)
    # hist, bin_edges = np.histogram(gene, density=True)
    # print hist
    # print bin_edges

    # plt.hist(gene, bins=20)
    # plt.title("hist")
    # plt.show()
    # raw_input()
