import os
import cv2
import glob
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from nets import mcnn

def visualization(img, density_map):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(density_map)

    plt.tight_layout()
    plt.show()

tf.app.flags.DEFINE_string(
    'model_path', None,
    'specify the model path for forecasting')
tf.app.flags.DEFINE_string(
    'image_path', None,
    'specify the image path for forecasting')

CHANNELS = 1
FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.model_path is None:
        raise ValueError('you must supply the model path with --model_path=<...>')
    if FLAGS.image_path is None:
        raise ValueError('you must supply the image path with --image_path=<...>')

    model = mcnn.McnnModel()
    model_path = FLAGS.model_path
    image_path = FLAGS.image_path
    
    img_input = tf.placeholder(tf.float32, shape=(512, 512, CHANNELS))
    batch_img = tf.expand_dims(img_input, axis=0)
    pred_dmap = model.net(batch_img, is_training=False)

    dmap_loss = 0.0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        flag = 1 if CHANNELS == 3 else 0
        img_data = cv2.imread(image_path, flag)
        img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB) if flag != 0 else img_data
        img = cv2.resize(img, (512, 512))
        img = (img.astype(np.float) - 127.5) / 128.0
        img = np.expand_dims(img, axis = -1) if len(img.shape) == 2 else img
        r_pred_dmap = sess.run([pred_dmap], feed_dict={img_input: img})

        dmap = np.array(r_pred_dmap[0])
        dmap = np.reshape(dmap, [128, 128])
        img = cv2.resize(img_data, (512, 512))
        print('pred counts: ', np.sum(dmap))
        visualization(img, dmap)

if __name__ == '__main__':
    tf.app.run()
