from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import os.path

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64s_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

data_dir = 'data'
cat_counts = np.load('%s/cat_counts.npy' % data_dir)
cats = np.load('%s/cats.npy' % data_dir).tolist()

out_dir = "tfrs"
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
  
image_dir = "profile_images"
  
count = 0

lis = os.listdir(image_dir)
# print o
cat_id = 0 
for li in lis:
  cat_dir = '%s/%s' % (image_dir, li)
  if os.path.isdir(cat_dir):
    print li
    ims = os.listdir(cat_dir)
    nIms = len(ims)
    im_id = 0
    for im in ims:
      out_file = "%s/%04d_%04d.tfrecord" % (out_dir,cat_id,im_id)
      if os.path.isfile(out_file):
        sys.stdout.write(':')
        sys.stdout.flush()
      else:
        im_path = '%s/%s' % (cat_dir, im)
        image = np.array(Image.open(im_path))
        height = int(image.shape[0])
        width = int(image.shape[1])
        image_raw = image.tostring()
        
        compress = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(out_file, options=compress)
        example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(height),
          'width': _int64_feature(width),
          'image_raw': _bytes_feature(image_raw),
          'cat_id': _int64_feature(cat_id),
        }))
        writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.write('.')
        sys.stdout.flush()
      im_id += 1
    cat_id += 1
print 'done'      
