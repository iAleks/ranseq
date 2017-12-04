import os
from model import Model
import tensorflow as tf
import hyperparams as hyp
from tensorflow.python import debug as tf_debug

def main(_):
    print "#"*50
    print hyp.name
    print "#"*50
    checkpoint_dir_ = os.path.join("checkpoint", hyp.name)
    log_dir_ = os.path.join("log", hyp.name)
    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)
        
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True

    if hyp.do_profile:
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()
        opts['min_cpu_micros'] = 5000
        opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                              trace_steps=range(10, 20),
                                              dump_steps=[20]) as pctx:
            pctx.add_auto_profiling('scope', opts, [19])
            with tf.Session(config=c) as sess:
                model = Model(sess,
                              checkpoint_dir=checkpoint_dir_,
                              log_dir=log_dir_
                )
                model.go()
    else:
        # g = tf.Graph()
        # with g.as_default():
        tf.set_random_seed(1)
        with tf.Session(config=c) as sess:
            model = Model(sess,
                          checkpoint_dir=checkpoint_dir_,
                          log_dir=log_dir_
            )
            model.go()

if __name__ == '__main__':
    tf.app.run()
    
