import os
import glob

import tensorflow as tf

# here 4 is the shrink size
RESIZED_IMAGE_SHAPE = [512, 512]
DENSITY_MAP_SIZE = (RESIZED_IMAGE_SHAPE[0] / 4) * (RESIZED_IMAGE_SHAPE[1] / 4)

def get_dataset(dataset_dir, FLAGS):
    example_fmt = {
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='RAW'),
        'image/density_map/shape': tf.FixedLenFeature([2], tf.int64),
        'image/density_map/data': tf.FixedLenFeature([DENSITY_MAP_SIZE], dtype=tf.float32),
    }

    def parse_fn(example):
        parsed = tf.parse_single_example(example, example_fmt)
        image = tf.decode_raw(parsed['image/encoded'], out_type=tf.uint8)
        image_shape = parsed['image/shape']
        density_map = parsed['image/density_map/data']
        density_map_shape = parsed['image/density_map/shape']

        image = tf.reshape(image, image_shape)
        density_map = tf.reshape(density_map, density_map_shape)
        image = tf.divide(tf.subtract(tf.cast(image, tf.float32), 127.5), 128.0)
        return image, density_map 

    data_sources = dataset_dir + '/*.tfrecord'
    data_files = glob.glob(data_sources)
    dataset = tf.data.TFRecordDataset(data_files)
    # dataset = dataset.map(map_func=parse_fn)
    # dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(FLAGS.shuffle_buffer_size, -1))
    # dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=FLAGS.batch_size))
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_preprocessing_threads)
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
    return dataset

