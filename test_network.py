import os
import cv2
import glob
import math
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
import scipy.io as sio

from nets import mcnn

tf.app.flags.DEFINE_string(
    'model_path', None,
    'specify the model path for testing')

CHANNELS = 3
FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.model_path is None:
        raise ValueError('you must supply the model path with --model_path=<...>')

    model = mcnn.McnnModel()
    model_path = FLAGS.model_path
    root_dir = './data/shtech/part_B_final/test_data/'
    root_dir = '/Users/wangpeng/Work/rongyi/crowd_counting/data/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/'
    image_path = root_dir + 'images/'
    annt_path = root_dir + 'ground_truth/'
    image_files = glob.glob(image_path + '*.jpg')
    
    img_input = tf.placeholder(tf.float32, shape=(512, 512, CHANNELS))
    batch_img = tf.expand_dims(img_input, axis=0)
    pred_dmap = model.net(batch_img, is_training=False)

    y_true = []
    y_pred = []
    dmap_loss = 0.0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        flag = 1 if CHANNELS == 3 else 0
        for i, img_file in enumerate(image_files):
            img = cv2.imread(img_file, flag)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if flag != 0 else img
            img = cv2.resize(img, (512, 512))
            img = (img.astype(np.float) - 127.5) / 128.0
            img = np.expand_dims(img, axis = -1) if len(img.shape) == 2 else img
            img_name = os.path.basename(img_file)[:-4]
            annt_file = annt_path + 'GT_' + img_name + '.mat'
            annt_data = sio.loadmat(annt_file)
            annts = annt_data['image_info'][0][0][0][0][0]
            r_pred_dmap = sess.run([pred_dmap], feed_dict={img_input: img})

            y_true.append(annts.shape[0])
            y_pred.append(int(np.sum(r_pred_dmap)))
            print('%s: true= %f, pred= %f' % (os.path.basename(img_file), y_true[-1], y_pred[-1]))

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))

if __name__ == '__main__':
    tf.app.run()
