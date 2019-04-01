import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class VggModel(object):
    """
    ref to CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    """
    def __init__(self):
        pass

    def net(self,
            inputs,
            weight_decay = 0.005,
            is_training = True,
            scope = 'vgg_16'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            normalizer_params={'is_training': is_training}):
            return self.nets_v1(inputs, is_training, scope)

    def nets_v1(self, inputs, is_training, scope='vgg_16'):
        end_points = {}
        with tf.variable_scope(scope, 'vgg_16'):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            end_points['block4'] = net

        # density_map = self.backend(net, end_points, 'backend')
        # density_map = self.backend_inception(net, end_points, 'backend')
        # density_map = self.backend_inception_v2(net, end_points, 'backend')
        density_map = self.backend_inception_v3(net, end_points, 'backend')
        return density_map

    def backend(self, inputs, end_points, scope='backend'):
        with tf.variable_scope(scope, 'backend'):
            net = slim.conv2d(inputs, 512, [3, 3], rate=2, scope='conv1')
            net = slim.conv2d(net, 512, [3, 3], rate=2, scope='conv2')
            net = slim.conv2d(net, 512, [3, 3], rate=2, scope='conv3')
            net = slim.conv2d(net, 256, [3, 3], rate=2, scope='conv4')
            net = slim.conv2d(net, 128, [3, 3], rate=2, scope='conv5')
            net = slim.conv2d(net, 64, [3, 3], rate=2, scope='conv6')
            density_map = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    def backend_inception(self, inputs, end_points, scope='backend'):
        with tf.variable_scope(scope, 'backend'):
            net = slim.conv2d(inputs, 256, [3, 3], scope='conv1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2')
            net = self.inception(net, 'inception_block')
            density_map = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    def backend_inception_v2(self, inputs, end_points, scope='backend'):
        with tf.variable_scope(scope, 'backend'):
            net = slim.conv2d(inputs, 256, [3, 3], normalizer_fn=slim.batch_norm, scope='conv1')
            net = slim.conv2d(net, 256, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
            net = slim.conv2d(net, 128, [3, 3], normalizer_fn=slim.batch_norm, scope='conv3')
            net = self.inception(net, 'inception_block')
            density_map = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map
    
    def backend_inception_v3(self, inputs, end_points, scope='backend'):
        with tf.variable_scope(scope, 'backend'):
            net = self.inception(inputs, 'inception_block_1')
            net = self.inception(net, 'inception_block_2')
            net = self.inception(net, 'inception_block_3')
            density_map = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map


    def inception(self, inputs, scope):
        with tf.variable_scope(scope, 'inception_block'):
            with tf.variable_scope('branch1x1'):
                branch1x1 = slim.conv2d(inputs, 64, [1, 1], scope='conv1')
            with tf.variable_scope('max_pool_1x1'):
                max_pool_1x1 = slim.max_pool2d(inputs, [3, 3], stride=1, padding='SAME', scope='max_pool')
                max_pool_1x1 = slim.conv2d(max_pool_1x1, 32, [1, 1], scope='conv1')
            with tf.variable_scope('branch3x3'):
                branch3x3 = slim.conv2d(inputs, 64, [1, 1], scope='conv1')
                branch3x3 = slim.conv2d(branch3x3, 64, [3, 3], scope='conv2')
            with tf.variable_scope('branch5x5'):
                branch5x5 = slim.conv2d(inputs, 48, [1, 1], scope='conv1')
                branch5x5 = slim.conv2d(branch5x5, 48, [3, 3], scope='conv2')
                branch5x5 = slim.conv2d(branch5x5, 32, [3, 3], scope='conv3')
            nets = tf.concat([branch1x1, max_pool_1x1, branch3x3, branch5x5], axis=-1, name='filter_concat')
            return nets
    
    def losses(self, y_pred, y_true, scope='VggLoss'):
        """
        y_pred with shape [batch_size, height, width, 1]
        y_true with shape [batch_size, height, width]
        """
        # return mcnn_square_loss(y_pred, y_true, scope)
        return mcnn_sum_square_loss_v2(y_pred, y_true, scope)

    def get_feature_map_height(self, height):
        return int(height / 8)

    def get_feature_map_width(self, width):
        return int(width / 8)

def mcnn_square_loss(y_pred, y_true, scope='VggLoss'):
    with tf.variable_scope(scope, 'VggLoss'):
        batch_size = y_pred.shape[0]
        y_pred = tf.reshape(y_pred, shape=[batch_size, -1])
        y_true = tf.reshape(y_true, shape=[batch_size, -1])
        d = tf.squared_difference(y_pred, y_true, name='sqr_d')
        d_sum = 0.5 * tf.reduce_sum(d, axis=1, name='sum')
        loss = tf.reduce_mean(d_sum, name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        y_pred_count = tf.reduce_sum(y_pred, axis=1)
        y_true_count = tf.reduce_sum(y_true, axis=1)
        abs_d = tf.abs(y_pred_count - y_true_count)
        sqr_d = tf.squared_difference(y_pred_count, y_true_count)
        mae = tf.reduce_mean(abs_d)
        mse = tf.reduce_mean(sqr_d)
        tf.summary.scalar('MAE', mae)
        tf.summary.scalar('RMSE', tf.sqrt(mse))

        return loss

def mcnn_sum_square_loss_v2(y_pred, y_true, scope='VggLoss'):
    with tf.variable_scope(scope, 'VggLoss'):
        batch_size = y_pred.shape[0]
        y_pred_sum = tf.reduce_sum(y_pred, axis=[1, 2, 3])
        y_true_sum = tf.reduce_sum(y_true, axis=[1, 2])
        d = 0.5 * tf.squared_difference(y_pred_sum, y_true_sum, name='square_diff_1')
        square_loss = tf.reduce_mean(d)

        y_pred = tf.reshape(y_pred, shape=[batch_size, -1])
        y_true = tf.reshape(y_true, shape=[batch_size, -1])
        d = tf.squared_difference(y_pred, y_true, name='square_diff_2')
        sum_mean = 0.5 * tf.reduce_sum(d, axis = 1)
        pixel_square_loss = tf.reduce_mean(sum_mean)

        beta = 0.05
        loss = tf.add(beta * square_loss, (1 - beta) * pixel_square_loss, name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        tf.summary.scalar('PIXEL_RMSE', tf.sqrt(pixel_square_loss))
        tf.summary.scalar('RMSE', tf.sqrt(square_loss))

        return loss




