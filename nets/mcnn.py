import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class McnnModel(object):
    def __init__(self):
        pass

    def net(self,
            inputs,
            weight_decay = 0.005,
            is_training = True,
            scope = 'MCNN'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            normalizer_params={'is_training': is_training}):
            # return self.nets_v1(inputs, is_training, scope)
            # return self.nets_v2(inputs, is_training, scope)
            # return self.nets_v3(inputs, is_training, scope)
            return self.nets_v4(inputs, is_training, scope)

    def nets_v1(self, inputs, is_training, scope):
        with tf.variable_scope(scope, 'MCNN'):
            with tf.variable_scope('LINE1'):
                cnn_line1 = slim.conv2d(inputs, 16, [9, 9], scope='conv1')
                cnn_line1 = slim.max_pool2d(cnn_line1, [2, 2], scope='pool1')
                cnn_line1 = slim.conv2d(cnn_line1, 32, [7, 7], scope='conv2')
                cnn_line1 = slim.max_pool2d(cnn_line1, [2, 2], scope='pool2')
                cnn_line1 = slim.conv2d(cnn_line1, 16, [7, 7], scope='conv3')
                cnn_line1 = slim.conv2d(cnn_line1, 8, [7, 7], scope='conv4')
            with tf.variable_scope('LINE2'):
                cnn_line2 = slim.conv2d(inputs, 20, [7, 7], scope='conv1')
                cnn_line2 = slim.max_pool2d(cnn_line2, [2, 2], scope='pool1')
                cnn_line2 = slim.conv2d(cnn_line2, 40, [5, 5], scope='conv2')
                cnn_line2 = slim.max_pool2d(cnn_line2, [2, 2], scope='pool2')
                cnn_line2 = slim.conv2d(cnn_line2, 20, [5, 5], scope='conv3')
                cnn_line2 = slim.conv2d(cnn_line2, 10, [5, 5], scope='conv4')
            with tf.variable_scope('LINE3'):
                cnn_line3 = slim.conv2d(inputs, 24, [5, 5], scope='conv1')
                cnn_line3 = slim.max_pool2d(cnn_line3, [2, 2], scope='pool1')
                cnn_line3 = slim.conv2d(cnn_line3, 48, [3, 3], scope='conv2')
                cnn_line3 = slim.max_pool2d(cnn_line3, [2, 2], scope='pool2')
                cnn_line3 = slim.conv2d(cnn_line3, 24, [3, 3], scope='conv3')
                cnn_line3 = slim.conv2d(cnn_line3, 12, [3, 3], scope='conv4')
            feature_map = tf.concat([cnn_line1, cnn_line2, cnn_line3], axis = -1, name='concat_feature_map')

            ##########################for model_v8###############################
            # feature_map = slim.conv2d(feature_map, 64, [3, 3], scope='fuse_conv')
            ##########################for model_v8###############################

            density_map = slim.conv2d(feature_map, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    # model_v14
    def nets_v2(self, inputs, is_training, scope):
        with tf.variable_scope(scope, 'MCNN'):
            nets = slim.conv2d(inputs, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv1')
            nets = slim.conv2d(nets, 64, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
            nets = slim.max_pool2d(nets, [2, 2], stride=2, scope='pool1')
            nets = self.inception(nets, scope='inception_block_1')
            nets = self.inception(nets, scope='inception_block_2')
            nets = slim.max_pool2d(nets, [2, 2], stride=2, scope='pool2')
            nets = self.inception(nets, scope='inception_block_3')
            nets = self.inception(nets, scope='inception_block_4')
            density_map = slim.conv2d(nets, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    # model_v15
    def nets_v3(self, inputs, is_training, scope):
        with tf.variable_scope(scope, 'MCNN'):
            nets = slim.conv2d(inputs, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv1')
            nets = slim.max_pool2d(nets, [2, 2], stride=2, scope='pool1')
            nets = self.inception_v2(nets, scope='inception_block_1')
            nets = self.inception_v2(nets, scope='inception_block_2')
            nets = slim.max_pool2d(nets, [2, 2], stride=2, scope='pool2')
            nets = self.inception_v2(nets, scope='inception_block_3')
            density_map = slim.conv2d(nets, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    # model_v16 use this nets
    def nets_v4(self, inputs, is_training, scope):
        with tf.variable_scope(scope, 'MCNN'):
            nets = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
            nets = slim.conv2d(nets, 32, [3, 3], scope='conv2')
            nets = slim.conv2d(nets, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv3')
            nets = slim.max_pool2d(nets, [2, 2], stride=2, scope='pool1')

            with tf.variable_scope('LINE1'):
                branch7x7 = slim.conv2d(nets, 16, [1, 1], scope='conv1')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], scope='conv2_1')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], scope='conv2_2')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2_3')
                branch7x7 = slim.max_pool2d(branch7x7, [2, 2], stride=2, scope='pool1')

                branch7x7 = slim.conv2d(branch7x7, 16, [1, 1], scope='conv3')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], scope='conv4_1')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], scope='conv4_2')
                branch7x7 = slim.conv2d(branch7x7, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv4_3')

                branch7x7 = slim.conv2d(branch7x7, 8, [1, 1], scope='conv5')
                branch7x7 = slim.conv2d(branch7x7, 8, [3, 3], scope='conv6_1')
                branch7x7 = slim.conv2d(branch7x7, 8, [3, 3], scope='conv6_2')
                branch7x7 = slim.conv2d(branch7x7, 8, [3, 3], scope='conv6_3')
            with tf.variable_scope('LINE2'):
                branch5x5 = slim.conv2d(nets, 20, [1, 1], scope='conv1')
                branch5x5 = slim.conv2d(branch5x5, 40, [3, 3], scope='conv2_1')
                branch5x5 = slim.conv2d(branch5x5, 40, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2_2')
                branch5x5 = slim.max_pool2d(branch5x5, [2, 2], stride=2, scope='pool1')

                branch5x5 = slim.conv2d(branch5x5, 20, [1, 1], scope='conv3')
                branch5x5 = slim.conv2d(branch5x5, 20, [3, 3], scope='conv4_1')
                branch5x5 = slim.conv2d(branch5x5, 20, [3, 3], normalizer_fn=slim.batch_norm, scope='conv4_2')

                branch5x5 = slim.conv2d(branch5x5, 10, [1, 1], scope='conv5')
                branch5x5 = slim.conv2d(branch5x5, 10, [3, 3], scope='conv6_1')
                branch5x5 = slim.conv2d(branch5x5, 10, [3, 3], scope='conv6_2')
            with tf.variable_scope('LINE3'):
                branch3x3 = slim.conv2d(nets, 48, [3, 3], normalizer_fn=slim.batch_norm, scope='conv1')
                branch3x3 = slim.max_pool2d(branch3x3, [2, 2], stride=2, scope='pool1')

                branch3x3 = slim.conv2d(branch3x3, 24, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
                branch3x3 = slim.conv2d(branch3x3, 12, [3, 3], scope='conv3')

            feature_map = tf.concat([branch7x7, branch5x5, branch3x3], axis = -1, name='concat_feature_map')
            density_map = slim.conv2d(feature_map, 1, [1, 1], activation_fn=None, scope='density_map')
            return density_map

    def inception(self, inputs, scope):
        with tf.variable_scope(scope, 'inception_block'):
            with tf.variable_scope('branch1x1'):
                branch1x1 = slim.conv2d(inputs, 64, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
            with tf.variable_scope('max_pool_1x1'):
                max_pool_1x1 = slim.max_pool2d(inputs, [3, 3], stride=1, padding='SAME', scope='max_pool')
                max_pool_1x1 = slim.conv2d(max_pool_1x1, 32, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
            with tf.variable_scope('branch3x3'):
                branch3x3 = slim.conv2d(inputs, 64, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
                branch3x3 = slim.conv2d(branch3x3, 64, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
            with tf.variable_scope('branch5x5'):
                branch5x5 = slim.conv2d(inputs, 48, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
                branch5x5 = slim.conv2d(branch5x5, 48, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
                branch5x5 = slim.conv2d(branch5x5, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv3')
            nets = tf.concat([branch1x1, max_pool_1x1, branch3x3, branch5x5], axis=-1, name='filter_concat')
            return nets
    
    def inception_v2(self, inputs, scope):
        """
        for nets_v3
        """
        with tf.variable_scope(scope, 'inception_block'):
            with tf.variable_scope('branch1x1'):
                branch1x1 = slim.conv2d(inputs, 32, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
            with tf.variable_scope('max_pool_1x1'):
                max_pool_1x1 = slim.max_pool2d(inputs, [3, 3], stride=1, padding='SAME', scope='max_pool')
                max_pool_1x1 = slim.conv2d(max_pool_1x1, 32, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
            with tf.variable_scope('branch3x3'):
                branch3x3 = slim.conv2d(inputs, 64, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
                branch3x3 = slim.conv2d(branch3x3, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
            with tf.variable_scope('branch5x5'):
                branch5x5 = slim.conv2d(inputs, 48, [1, 1], normalizer_fn=slim.batch_norm, scope='conv1')
                branch5x5 = slim.conv2d(branch5x5, 48, [3, 3], normalizer_fn=slim.batch_norm, scope='conv2')
                branch5x5 = slim.conv2d(branch5x5, 32, [3, 3], normalizer_fn=slim.batch_norm, scope='conv3')
            nets = tf.concat([branch1x1, max_pool_1x1, branch3x3, branch5x5], axis=-1, name='filter_concat')
            return nets


    def losses(self, y_pred, y_true, scope='McnnLoss'):
        """
        y_pred with shape [batch_size, height, width, 1]
        y_true with shape [batch_size, height, width]
        """
        # return mcnn_square_loss(y_pred, y_true, scope)
        # return mcnn_smooth_abs(y_pred, y_true, scope)
        # return mcnn_sum_square_loss(y_pred, y_true, scope)
        # return mcnn_mesa_loss(y_pred, y_true, scope)
        return mcnn_sum_square_loss_v2(y_pred, y_true, scope)

    def get_feature_map_height(self, height):
        return int(height / 4)

    def get_feature_map_width(self, width):
        return int(width / 4)


def mcnn_square_loss(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
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

def mcnn_smooth_abs(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
        batch_size = y_pred.shape[0]
        y_pred = tf.reshape(y_pred, shape=[-1])
        y_true = tf.reshape(y_true, shape=[-1])
        abs_diff = tf.abs(y_pred - y_true)
        min_diff = tf.minimum(abs_diff, 1)
        r = 0.5 * ((abs_diff - 1) * min_diff + abs_diff)
        total = tf.reduce_sum(r, name='total')
        loss = tf.divide(total, tf.cast(batch_size, tf.float32), name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        y_pred_count = tf.reduce_sum(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        y_true_count = tf.reduce_sum(tf.reshape(y_true, [batch_size, -1]), axis=1)
        abs_diff = tf.abs(y_pred_count - y_true_count)
        mae = tf.reduce_mean(abs_diff)
        tf.summary.scalar('MAE', mae)

        return loss

def mcnn_sum_square_loss(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
        batch_size = y_pred.shape[0]
        y_pred_sum = tf.reduce_sum(y_pred, axis=[1, 2, 3])
        y_true_sum = tf.reduce_sum(y_true, axis=[1, 2])
        d = 0.5 * tf.squared_difference(y_pred_sum, y_true_sum, name='diff')
        square_loss = tf.reduce_mean(d)

        y_pred = tf.reshape(y_pred, shape=[batch_size, -1])
        y_true = tf.reshape(y_true, shape=[batch_size, -1])
        d = tf.abs(y_pred - y_true)
        sum_mean = tf.reduce_sum(d, axis = 1)
        pixel_abs_loss = tf.reduce_mean(sum_mean)

        lamda = 1.0
        loss = tf.add(square_loss, lamda * pixel_abs_loss, name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        tf.summary.scalar('PIXEL_MAE', pixel_abs_loss)
        tf.summary.scalar('RMSE', tf.sqrt(square_loss))

        return loss

def mcnn_sum_square_loss_v2(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
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

        beta = 0.01
        loss = tf.add(beta * square_loss, (1 - beta) * pixel_square_loss, name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        tf.summary.scalar('PIXEL_RMSE', tf.sqrt(pixel_square_loss))
        tf.summary.scalar('RMSE', tf.sqrt(square_loss))

        return loss

def max_subarray(array):
    size = len(array)
    max_ending_here = array[0]
    max_so_far = max_ending_here
    start = 0
    end = 0
    indicate = 1 if max_ending_here < 0 else 0
    for i in range(1, size):
        x = array[i]
        max_ending_here = max(x, max_ending_here + x)
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = indicate
            end = i
        if max_ending_here < 0:
            indicate = i + 1
    return start, end, max_so_far

def max_subtensor(A):
    A = tf.reshape(A, [-1])
    size = A.shape[0]
    max_ending_here = -1
    max_so_far = -1e9
    start = 0
    end = 0
    indicate = 0
    for i in range(size):
        x = A[i]
        max_ending_here = tf.maximum(x, max_ending_here + x)

        def f1():
            return indicate, i, max_ending_here
        def f2():
            return start, end, max_so_far
        start, end, max_so_far = tf.cond(max_ending_here > max_so_far, f1, f2)

        indicate = tf.cond(max_ending_here < 0, lambda: i + 1, lambda: indicate)
    return start, end, max_so_far

def mcnn_mesa_loss(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
        dmap_shape = y_true.get_shape().as_list()
        y_pred = tf.reshape(y_pred, dmap_shape)
        d = tf.subtract(y_pred, y_true)

        batch_size, height, width = dmap_shape
        max_sums = np.zeros(batch_size)
        retangles = np.zeros((batch_size, 4), dtype=np.int32)
        for left in range(width):
            col_temp_sum = np.zeros((batch_size, height))
            for right in range(left, width):
                col_temp_sum += d[:, :, right]
                for i in range(batch_size):
                    start, end, max_sum = max_subtensor(col_temp_sum[i, :])
                    start_2, end_2, max_sum_2 = max_subtensor(0 - col_temp_sum[i, :])
                    start, end, max_sum = tf.cond(max_sum < max_sum_2,
                                                  lambda: (start_2, end_2, max_sum_2),
                                                  lambda: (start, end, max_sum))
                    m_ = max_sums[i]
                    l_ = retangles[i][0]
                    r_ = retangles[i][1]
                    s_ = retangles[i][2]
                    e_ = retangles[i][3]
                    m_, l_, r_, s_, e_ = tf.cond(max_sum > m_,
                                                 lambda: (max_sum, left, right, start, end),
                                                 lambda: (m_, l_, r_, s_, e_))
                    max_sums[i] = m_
                    retangles[i][0] = l_
                    retangles[i][1] = r_
                    retangles[i][3] = s_
                    retangles[i][4] = e_

        indices = []
        values = []
        for i in range(batch_size):
            for j in range(retangles[i][3], retangles[i][4] + 1):
                for k in range(retangles[i][0], retangles[i][1] + 1):
                    indices.append([i, j, k])
                    values.append(1.0)
        sparse_weights = tf.SparseTensor(indices, values, dmap_shape)
        weights = tf.sparse_tensor_to_dense(sparse_weights, default_value=0)
        weighted_d = weights * d
        sum_d = tf.reduce_sum(weighted_d, axis=[1, 2])
        loss = tf.reduce_mean(tf.abs(sum_d))
        tf.losses.add_loss(loss)
            
        # summary
        y_pred_count = tf.reduce_sum(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        y_true_count = tf.reduce_sum(tf.reshape(y_true, [batch_size, -1]), axis=1)
        abs_diff = tf.abs(y_pred_count - y_true_count)
        mae = tf.reduce_mean(abs_diff)
        tf.summary.scalar('MAE', mae)

        return loss


def _test_max_subarray():
    # lst = [-100, 1, 5, -3, 10]
    lst = [100, 1, 5, -30, 10]
    start, end, max_sum = max_subarray(lst)
    print('start= %i, end= %i, max= %f' % (start, end, max_sum))

if __name__ == '__main__':
    _test_max_subarray()







