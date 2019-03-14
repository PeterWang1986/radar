import numpy as np
import tensorflow as tf

import tf_utils
from datasets import dataset_factory
from nets import mcnn

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', './tmp/models/',
    'store checkponts and event logs for training')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'the number of parallel readers that read data from file sytem')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'the number of threads used to create the batches')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'the frequency with which logs are print')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'the frequency with which summaries are saved, in seconds')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'the frequency with which the model is saved, in seconds')
tf.app.flags.DEFINE_integer(
    'max_number_steps', None,
    'the maximum number of training steps')

# for learning rate
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'the minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005,
    'the weight decay on the model weights')
tf.app.flags.DEFINE_string(
    'dataset_name', 'shtech_part_B',
    'the name of the dataset to load')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'the directory where the dataset files are stored')
tf.app.flags.DEFINE_integer(
    'batch_size', 16,
    'the number of samples in each batch')
tf.app.flags.DEFINE_integer(
    'shuffle_buffer_size', 160,
    'buffer size for shuffleing(e.g. batch_size * n)')
tf.app.flags.DEFINE_integer(
    'prefetch_buffer_size', 32,
    'buffer size for prefetching(e.g. batch_size * 2)')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'the path to a checkpoint from which to fine-tune')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'comma-separated list of scopes of variables to exclude when restoring from a checkpoint')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'comma-separated list of scopes to filter the set of variables to train')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')


CHANNELS = 3
DATA_SIZE = 3000
RESIZED_IMAGE_SHAPE = [512, 512]
FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.dataset_dir is None:
        raise ValueError('you must supply the dataset directory with --dataset_dir=')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        global_step = tf.train.create_global_step()
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS)
        iterator = dataset.make_one_shot_iterator()

        model = mcnn.McnnModel()
        h, w = RESIZED_IMAGE_SHAPE
        f_h = model.get_feature_map_height(h)
        f_w = model.get_feature_map_width(w)
        b_img, b_dmap = iterator.get_next()
        b_img = tf.reshape(b_img, [FLAGS.batch_size, h, w, CHANNELS])
        b_dmap = tf.reshape(b_dmap, [FLAGS.batch_size, f_h, f_w])
        pred_dmap = model.net(b_img)
        sqr_loss = model.losses(pred_dmap, b_dmap)
        tf.summary.image('image', b_img)

        # add summaries for loss
        total_loss = slim.losses.get_total_loss()
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        learning_rate = tf_utils.create_learning_rate(FLAGS, DATA_SIZE, global_step)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, name='Momentum')
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1.0)

        variables_to_train = tf_utils.get_variables_to_train(FLAGS)
        train_op = slim.learning.create_train_op(total_loss, optimizer,
                                                variables_to_train=variables_to_train)
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            init_fn = tf_utils.get_init_fn(FLAGS),
                            summary_op=summary_op,
                            save_summaries_secs=FLAGS.save_summaries_secs,
                            log_every_n_steps=FLAGS.log_every_n_steps,
                            number_of_steps=FLAGS.max_number_steps,
                            saver=saver)

if __name__ == '__main__':
    tf.app.run()


















