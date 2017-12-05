import tensorflow as tf
import hyperparams as hyp
import batcher
from utils import print_shape

def get_inputs(model):
    model.gene_t = tf.ones((hyp.B, hyp.N, 1))
    model.gene_v = tf.ones((hyp.B, hyp.N, 1))
    model.cat_t = tf.ones((hyp.B), tf.int64)
    model.cat_v = tf.ones((hyp.B), tf.int64)

    
    if not hyp.do_batch_balance:
        (model.gene_t,
         model.cat_t) = batcher.batch(hyp.dataset_t,
                                      hyp.B,
                                      aug=hyp.aug_train,
                                      shuffle=hyp.shuffle_train)
        (model.gene_v,
         model.cat_v) = batcher.batch(hyp.dataset_v,
                                      hyp.B,
                                      aug=hyp.aug_val,
                                      shuffle=hyp.shuffle_val)
    else:
        genes = []
        cats = []
        for g in range(hyp.nCats):
            (gene, cat) = batcher.batch('%s_%d.txt' % (hyp.dataset_t[:-4], g),
                                        hyp.B,
                                        shuffle=hyp.shuffle_train)
            genes.append(gene)
            cats.append(cat)
        model.gene_t = tf.concat([genes], axis=0)
        model.cat_t = tf.concat([cats], axis=0)

        genes = []
        cats = []
        for g in range(hyp.nCats):
            (gene, cat) = batcher.batch('%s_%d.txt' % (hyp.dataset_t[:-4], g),
                                        hyp.B,
                                        aug=hyp.aug_train,
                                        shuffle=hyp.shuffle_train)
            genes.append(gene)
            cats.append(cat)
        print_shape(genes[0])
        model.gene_t = tf.concat(genes, axis=0)
        print_shape(model.gene_t)
        model.cat_t = tf.concat(cats, axis=0)
        print_shape(model.cat_t)

        genes = []
        cats = []
        for g in range(hyp.nCats):
            (gene, cat) = batcher.batch('%s_%d.txt' % (hyp.dataset_v[:-4], g),
                                        hyp.B,
                                        catid=g,
                                        aug=hyp.aug_val,
                                        shuffle=hyp.shuffle_val)
            genes.append(gene)
            cats.append(cat)
        model.gene_v = tf.concat(genes, axis=0)
        model.cat_v = tf.concat(cats, axis=0)
        # model.gene_v = tf.concat([gene for gene in genes], axis=0)
        # model.cat_v = tf.concat([cat for cat in cats], axis=0)
    
    model.train_inputs = [
        model.gene_t,
        model.cat_t,
    ]

    model.val_inputs = [
        model.gene_v,
        model.cat_v,
    ]
    
    return model.train_inputs, model.val_inputs

