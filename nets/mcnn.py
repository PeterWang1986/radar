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
                            biases_initializer=tf.zeros_initializer()):
            return self.nets_v1(inputs, is_training, scope)

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

    def losses(self, y_pred, y_true, scope='McnnLoss'):
        """
        y_pred with shape [batch_size, height, width, 1]
        y_true with shape [batch_size, height, width]
        """
        return mcnn_square_loss(y_pred, y_true, scope)
        # return mcnn_smooth_abs(y_pred, y_true, scope)
        # return mcnn_sum_square_loss(y_pred, y_true, scope)
        # return mcnn_mesa_loss(y_pred, y_true, scope)

    def get_feature_map_height(self, height):
        return int(height / 4)

    def get_feature_map_width(self, width):
        return int(width / 4)


def mcnn_square_loss(y_pred, y_true, scope='McnnLoss'):
    with tf.variable_scope(scope, 'McnnLoss'):
        batch_size = y_pred.shape[0]
        y_pred = tf.reshape(y_pred, shape=[batch_size, -1])
        y_true = tf.reshape(y_true, shape=[batch_size, -1])
        diff = tf.squared_difference(y_pred, y_true, name='squared_difference')
        total_sum = tf.reduce_sum(diff, name='total_difference_sum')
        loss = tf.divide(total_sum, 2.0 * tf.cast(batch_size, tf.float32), name='loss_value')
        tf.losses.add_loss(loss)

        # summary
        y_pred_count = tf.reduce_sum(y_pred, axis=1)
        y_true_count = tf.reduce_sum(y_true, axis=1)
        abs_diff = tf.abs(y_pred_count - y_true_count)
        mae = tf.reduce_mean(abs_diff)
        tf.summary.scalar('MAE', mae)

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

        beta = 0.7
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
    """
    currently NOT work
    """
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







