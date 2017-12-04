import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import hyperparams as hyp

def load_weights(saver, sess):
    # load_resnet(sess)
    if hyp.total_init:
        print "\n\nTOTAL INIT\n\n"
        print hyp.total_init
        start_iter = load(saver, sess, hyp.total_init)
        if start_iter:
            print "loaded full model. resuming from iter %d" % start_iter
        else:
            print "could not find a full model. starting from scratch"
    else:
        # load weights for the subnets
        start_iter = 0
        inits = {"gene": hyp.gene_init}
        for part, init in inits.items():
            if init:
                iter = load_part(sess, part, init)
                if iter:
                    print "loaded %s at iter %d" % (init, iter)
                else:
                    print "could not find a checkpoint in " % init
    return start_iter

def save(saver, sess, checkpoint_dir, step):
    model_name = "minuet.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)
    print("Saved a checkpoint: %s/%s-%d" % (checkpoint_dir, model_name, step))

def load(saver, sess, model):
    print("reading full checkpoint...")

    print "looking for %d vars... " % len(tf.trainable_variables())
    # for v in tf.trainable_variables():
    #     print 'name = {}'.format(v.value())
    
    model_dir = os.path.join("checkpoint", model)
    print "looking in %s" % model_dir
    ckpt = tf.train.get_checkpoint_state(model_dir)
    print ckpt
    start_iter = 0
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("...found %s " % ckpt.model_checkpoint_path)
        start_iter = int(ckpt_name[len("minuet-model")+1:])
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
    else:
        print("...ain't no full checkpoint here!")
    return start_iter

def load_part(sess, part, init):
    print "reading %s checkpoint..." % part
    init_dir = os.path.join("checkpoint", init)
    print init_dir
    ckpt = tf.train.get_checkpoint_state(init_dir)
    start_iter = 0
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("...found %s " % ckpt.model_checkpoint_path)
        start_iter = int(ckpt_name[len("minuet-model")+1:])
        model_vars = slim.get_model_variables()
        scope = "inference/%s/%sNet" % (part, part.title())
        print "loading %s" % scope
        my_vars = slim.get_variables_to_restore(include=[scope])
        vars_to_restore = set(model_vars).intersection(my_vars)
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(sess, os.path.join(init_dir, ckpt_name))
    else:
        print "...ain't no %s checkpoints here!" % part
    return start_iter

def load_resnet(sess):
    new_vars = []
    print "looking for %d vars... " % len(tf.trainable_variables())
    # for v in tf.trainable_variables():
    #     print 'name = {}'.format(v.value())
    for v in tf.trainable_variables():
        name = v.op.name
        # print name
        if name.find("dummy")==-1 and name.find("fc")==-1:
            print 'restoring %s' % name
            # print "ok!"
            # v = tf.contrib.framework.load_variable('.', name)
            new_vars.append(tf.Variable(v, name=name.replace('inference/depth/', '')))
            # new_vars.append(tf.Variable(v, name=("inference/depth/" + name)))
    # saver = tf.train.Saver(new_vars)
    # restorer = tf.train.Saver(var_list=tf.trainable_variables())
    restorer = tf.train.Saver(var_list=new_vars)
    restorer.restore(sess, "deeplab_resnet.ckpt")
    print "restored resnet weights!"
    start_iter = 0
    return start_iter





