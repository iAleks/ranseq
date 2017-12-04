import tensorflow as tf
import sys
import hyperparams as hyp

def read_and_decode(filename_queue):
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=compress)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'gene_raw': tf.FixedLenFeature([], tf.string),
            'cat_id': tf.FixedLenFeature([], tf.int64),
        })

    gene = tf.decode_raw(features['gene_raw'], tf.float32)
    cat = tf.cast(features['cat_id'], tf.int64)

    gene = tf.reshape(gene, [hyp.N, 1])
    cat = tf.reshape(cat, [])
    
    return (gene,cat)
