import os
import sys
import cv2
import glob
import random

import numpy as np
import tensorflow as tf

import scipy.io as sio
from scipy.spatial import distance_matrix

# from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from dataset_utils import int64_feature, float_feature, bytes_feature

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'ground_truth/'
DIRECTORY_IMAGES = 'images/'

# generate density map parameters
TOPK = 6
BETA = 1.0
G_KENERL_SIZE = 15
SHRINK_RATIO = 4
RAND_CROP_COUNT = 10

# TFRecords convertion parameters
SAMPLES_PER_FILES = 20

# resize image parameters
CHANNELS = 3
RESIZED_IMAGE_SHAPE = [512, 512]

def _resize_image(img, annts):
    height, width = img.shape[0], img.shape[1]
    y_ratio = float(RESIZED_IMAGE_SHAPE[0]) / height
    x_ratio = float(RESIZED_IMAGE_SHAPE[1]) / width
    annts = np.array(annts) * np.array([x_ratio, y_ratio])
    img = cv2.resize(img, dsize=(RESIZED_IMAGE_SHAPE[1], RESIZED_IMAGE_SHAPE[0]))
    return img, annts 

def _random_crop_with_points(img, annts):
    try_count = 0
    height = img.shape[0]
    width = img.shape[1]
    origin_area = float(height * width)
    while try_count < 100:
        y = np.random.randint(0, int(height / 2) + 1, 1)[0]
        x = np.random.randint(0, int(width / 2) + 1, 1)[0]
        h = np.random.randint(0, height - y + 1, 1)[0]
        w = np.random.randint(0, width - x + 1, 1)[0]
        if (h * w) / origin_area > 0.5:
            break
        try_count += 1

    if try_count < 100:
        new_annts = []
        for annt in annts:
            annt_x = annt[0] - x
            annt_y = annt[1] - y
            if annt_x >= 0 and annt_x < w and annt_y >= 0 and annt_y < h:
                new_annts.append([annt_x, annt_y])
        if len(new_annts) > 0:
            if len(img.shape) == 3:
                return img[y : y + h, x : x + w, :], np.array(new_annts)
            else:
                return img[y : y + h, x : x + w], np.array(new_annts)

    return img, annts

def _gen_density_map(img, annts, ratio):
    dmap_height = int(img.shape[0] / ratio)
    dmap_width = int(img.shape[1] / ratio)
    density_map = np.zeros([dmap_height, dmap_width])

    if annts.shape[0] < TOPK:
        for annt in annts:
            cur_density_map = np.zeros([dmap_height, dmap_width])
            x, y = int(annt[0] / ratio), int(annt[1] / ratio)
            x = dmap_width - 1 if x >= dmap_width else x
            y = dmap_height - 1 if y >= dmap_height else y
            cur_density_map[y][x] = 1
            cur_density_map = cv2.GaussianBlur(cur_density_map, (G_KENERL_SIZE, G_KENERL_SIZE), sigmaX=0)
            density_map += cur_density_map
    else:
        dis_mat = distance_matrix(annts, annts, p=2)
        sorted_index = np.argsort(dis_mat, axis = -1)
        for i, annt in enumerate(annts):
            cur_density_map = np.zeros([dmap_height, dmap_width])
            top_k = dis_mat[i][sorted_index[i][1:TOPK]]
            avg_dis = np.average(np.sqrt(top_k))
            x, y = int(annt[0] / ratio), int(annt[1] / ratio)
            x = dmap_width - 1 if x >= dmap_width else x
            y = dmap_height - 1 if y >= dmap_height else y
            radius = BETA * avg_dis
            cur_density_map[y][x] = 1
            cur_density_map = cv2.GaussianBlur(cur_density_map, (G_KENERL_SIZE, G_KENERL_SIZE), sigmaX=radius/ratio)
            density_map += cur_density_map

    return density_map

def _process_image(root_dir, img_file):
    """Process a image and annotation file.
    """
    # Read the image file.
    flag = 1 if CHANNELS == 3 else 0
    img = cv2.imread(img_file, flag)
    # flag = 1 indicate BGR mode in opencv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if flag != 0 else img

    img_name = os.path.basename(img_file)[:-4]
    annt_file = root_dir + '/' + DIRECTORY_ANNOTATIONS + 'GT_' + img_name + '.mat'
    annt_data = sio.loadmat(annt_file)
    annts = annt_data['image_info'][0][0][0][0][0]

    image_data = []
    image_shapes = []
    density_maps = []
    density_shapes = []
    for i in range(RAND_CROP_COUNT):
        croped_img, croped_annts = _random_crop_with_points(img, annts)
        croped_img, croped_annts = _resize_image(croped_img, croped_annts)
        density_map = _gen_density_map(croped_img, croped_annts, ratio = SHRINK_RATIO)
        img_raw = croped_img.tostring()
        image_data.append(img_raw)
        image_shapes.append(croped_img.shape)
        density_maps.append(density_map)
        density_shapes.append(density_map.shape)

    return image_data, image_shapes, density_maps, density_shapes

def _convert_to_example(image_data, image_shape, density_map, density_shape):
    """Build an Example proto for an image example.

    Args:
      image_data: image raw data(string)
      image_shape: shape of image
      density_map: density map which is generated from annotations
      density_shape: shape of density map
    Returns:
      Example proto
    """
    assert(image_shape[0] == RESIZED_IMAGE_SHAPE[0])
    assert(image_shape[1] == RESIZED_IMAGE_SHAPE[1])
    assert(density_shape[0] == RESIZED_IMAGE_SHAPE[0] / SHRINK_RATIO)
    assert(density_shape[1] == RESIZED_IMAGE_SHAPE[1] / SHRINK_RATIO)
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(image_shape[0]),
            'image/width': int64_feature(image_shape[1]),
            'image/channels': int64_feature(CHANNELS),
            'image/shape': int64_feature([image_shape[0], image_shape[1], CHANNELS]),
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(b'RAW'),

            'image/density_map/shape': int64_feature([density_shape[0], density_shape[1]]),
            'image/density_map/data': float_feature(density_map.flatten())
            }))
    return example

def _add_to_tfrecord(dataset_dir, img_file, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      img_file: Image file to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, image_shapes, density_maps, density_shapes = _process_image(dataset_dir, img_file)
    size = len(image_data)
    for i in range(size):
        example = _convert_to_example(image_data[i], image_shapes[i], density_maps[i], density_shapes[i])
        tfrecord_writer.write(example.SerializeToString())
    return size

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='shtech_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    # Dataset filenames, and shuffling.
    img_path = os.path.join(dataset_dir, DIRECTORY_IMAGES)
    image_files = glob.glob(img_path + '/*.jpg')
    if shuffling:
        random.shuffle(image_files)

    # Process dataset files.
    i = 0
    fidx = 0
    total_records = 0
    kSIZE = len(image_files)
    while i < kSIZE:
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < kSIZE and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(image_files)))
                sys.stdout.flush()

                img_file = image_files[i]
                records = _add_to_tfrecord(dataset_dir, img_file, tfrecord_writer)
                i += 1
                j += 1
                total_records += records
            fidx += 1

    print('\nFinished converting dataset!')
    print('total records: ', total_records)

def main(_):
    dataset_dir = '../data/shtech/part_A_final/train_data'
    output_dir = '../data/shtech/part_A_final/train_data/tfrecords_3k'
    run(dataset_dir, output_dir, shuffling=True)

if __name__ == '__main__':
    tf.app.run()
