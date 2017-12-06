from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import dtypes
import sys
import hyperparams as hyp
from genenet import *
from saverloader import *
from placeholders import *
from inputs import get_inputs

class Model(object):
    def __init__(self, sess,
                 checkpoint_dir=None,
                 log_dir=None):
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.build_model()

        # _t = training
        # _v = validation

    def build_model(self):
        # get a batch of inputs
        self.train_inputs, self.val_inputs = get_inputs(self)
        # get a batch of inputs
        self.placeholders = get_placeholders(self)

        # define a global step/iteration number
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        ## infer, and compute the loss
        # handle the tf complaint about no loss
        dummy_var = tf.Variable(0.0, name="dummy")
        self.loss = tf.identity(dummy_var-dummy_var)
        loss_dict, self.pred_cat, self.cat = self.inference(is_train=hyp.do_train, reuse=False)
        for loss in loss_dict.values():
            self.loss = self.loss + loss
        self.loss += tf.reduce_sum(slim.losses.get_regularization_losses())
        tf.summary.scalar('loss', self.loss)

        ## define a big summary op we can run
        self.summary = tf.summary.merge_all()
        
        ## define a saverloader
        self.saver = tf.train.Saver(max_to_keep=100)

    def go(self):
        ### setup
        start_time = time.time()
        if hyp.do_train:
            print("------ TRAINING ------")
            # on train iters, we will run the optimizer to minimize the total sum loss
            rate = tf.train.exponential_decay(hyp.lr, self.global_step, 1, 0.9999)
            optimizer = tf.train.AdamOptimizer(rate, beta1=0.9, beta2=0.999) \
                                .minimize(self.loss, global_step=self.global_step)
        else:
            print("------ TESTING ------")
            # nothing to do here yet
            mkdir('predictions')
            outfile = open('predictions/%s.txt' % hyp.name, 'w')
        print hyp.name

        ## logging
        if not (hyp.dataset_t==hyp.dataset_v):
            if not hyp.do_fast_logging:
                writer_t = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
            else:
                writer_t = tf.summary.FileWriter(self.log_dir + '/train', None)
        if not hyp.do_fast_logging:
            writer_v = tf.summary.FileWriter(self.log_dir + '/val', self.sess.graph)
        else:
            writer_v = tf.summary.FileWriter(self.log_dir + '/val', None)

        ## startup
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ## load weights from checkpoint, if we need to
        start_iter = load_weights(self.saver, self.sess)
        if not hyp.do_train:
            start_iter = 0

        print "OK! Ready to go. (Setup took %.1f)"  % (time.time() - start_time)
        start_time = time.time()
        
        for step in range(start_iter+1, hyp.max_iters+1):
            total_time = time.time()-start_time
            read_start_time = time.time()
            train_feed = feed_from(self.train_inputs,
                                   self.placeholders,
                                   self.sess)

            read_time = time.time()-read_start_time
            # ...and optimize
            iter_start_time = time.time()
            if hyp.do_train:
                _, loss_t, run_sum, global_step = self.sess.run([optimizer,
                                                                 self.loss,
                                                                 self.summary,
                                                                 self.global_step],
                                                                feed_dict=train_feed)
                # log if we need to
                if (step==start_iter+1 or np.mod(step, hyp.log_freq_t) == 0):
                    if not (hyp.dataset_t==hyp.dataset_v):
                        writer_t.add_summary(run_sum, step)
                        iter_time = time.time()-iter_start_time
                        print "%s; iter:[%4d/%4d]; time: %.1f; rtime: %.2f; itime: %.2f; loss_t: %.3f" % (hyp.name,
                                                                                                          step,
                                                                                                          hyp.max_iters,
                                                                                                          total_time,
                                                                                                          read_time,
                                                                                                          iter_time,
                                                                                                          loss_t)
            if not hyp.do_train or (step==start_iter+1 or np.mod(step, hyp.log_freq_v) == 0):
                iter_time = time.time()-iter_start_time
                if not hyp.do_train or not (hyp.dataset_t==hyp.dataset_v):
                    # on every val iteration, get a val batch...
                    val_feed = feed_from(self.val_inputs,
                                         self.placeholders,
                                         self.sess)
                    thingstorun = {}
                    thingstorun['summ'] = [self.summary]
                    thingstorun['loss'] = [self.loss]
                    thingstorun['pred'] = [self.pred_cat]
                    thingstorun['cat'] = [self.cat]
                    # ...and evaluate it
                    results = self.sess.run(thingstorun, feed_dict=val_feed)
                    run_sum = results['summ'][0]
                    loss_v = results['loss'][0]
                    pred = results['pred'][0]
                    cat = results['cat'][0]
                    writer_v.add_summary(run_sum, step)
                    print "%s; iter:[%4d/%4d]; time: %.1f; rtime: %.2f; itime: %.2f: loss_v: %.3f" % (hyp.name,
                                                                                                      step,
                                                                                                      hyp.max_iters,
                                                                                                      total_time,
                                                                                                      read_time,
                                                                                                      iter_time,
                                                                                                      loss_v)
                else:
                    writer_v.add_summary(run_sum, step)
                    print "%s; iter:[%4d/%4d]; time: %.1f; rtime: %.2f; itime: %.2f" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time)

                if not hyp.do_train:
                    # outfile.write('%d %d\n' % (step, pred[0]))
                    outfile.write('%d %d %d\n' % (step, pred[0], cat[0]))
                    
            if hyp.do_train and (np.mod(step, hyp.snap_freq) == 0):
                # save a checkpoint
                save(self.saver, self.sess, self.checkpoint_dir, step)

        if not hyp.do_train:
            outfile.close()

        coord.request_stop()
        coord.join(threads)
    
    def inference(self, is_train=True, reuse=False):
        loss_dict = {}
        with tf.variable_scope("inference"):
            gene_loss_dict, pred_cat, cat = GeneNet(self.gene,
                                                    self.cat,
                                                    is_train=is_train,
                                                    reuse=reuse)
            loss_dict.update(gene_loss_dict)
        return loss_dict, pred_cat, cat

