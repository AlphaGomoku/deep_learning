import tensorflow as tf


VERSION = "v.beta2"


initializer = tf.contrib.layers.xavier_initializer()

def input_layer(X, num_input, num_output, dropout_rate=None):
    global fc_layer_cnt
    fc_layer_cnt = 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))

def hidden_layer(X, num_input, num_output, dropout_rate=None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))

def output_layer(X, num_input, num_output, dropout_rate=None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.add(tf.matmul(X, W), b)

def conv_layer(X, shape, strides=[1, 1, 1, 1]):
    global conv_layer_cnt
    conv_layer_cnt += 1
    W = tf.get_variable("Conv_W" + str(conv_layer_cnt), shape=shape, initializer=initializer)
    return tf.nn.conv2d(X, W, strides=strides, padding="SAME")

def relu_layer(X):
    return tf.nn.relu(X)

def max_pooling_layer(X, ksize, strides, padding="SAME"):
    return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding)

def avg_pooling_layer(X, ksize, strides, padding="SAME"):
    return tf.nn.avg_pool(X, ksize=ksize, strides=strides, padding=padding)

def inception(X, depths):
    input_depth = X.get_shape()[-1]
    with tf.variable_scope('branch_0'):
        branch_0 = conv_layer(X, shape=[1, 1, input_depth, depths[0]])
    with tf.variable_scope('branch_1'):
        branch_1 = conv_layer(X, shape=[1, 1, input_depth, depths[1]])
        branch_1 = conv_layer(branch_1, shape=[3, 3, depths[1], depths[2]])
    with tf.variable_scope('branch_2'):
        branch_2 = conv_layer(X, shape=[1, 1, input_depth, depths[3]])
        branch_2 = conv_layer(branch_2, shape=[5, 5, depths[3], depths[4]])
    with tf.variable_scope('branch_3'):
        branch_3 = avg_pooling_layer(X, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1])
        branch_3 = conv_layer(branch_3, shape=[1, 1, input_depth, depths[5]])
    return tf.concat(3, [branch_0, branch_1, branch_2, branch_3])

def make_model(X, dropout_rate):
    # X's shape is [?, 225]

    end_points = {}

    # Construct Conv / ReLU / Pooling layers
    net = tf.reshape(X, shape=[-1, 15, 15, 1])  # [?, 15, 15, 1]
    end_points['input'] = net

    global conv_layer_cnt
    conv_layer_cnt = 0

    net = inception(net, [8, 6, 8, 8, 12, 4])  # [?, 15, 15, 32]
    end_points['inception_1'] = net

    net = inception(net, [16, 12, 16, 16, 24, 8])  # [?, 15, 15, 64]
    end_points['inception_2'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 7, 7, 64]
    end_points['pooling_2'] = net

    net = inception(net, [32, 24, 32, 32, 48, 16])  # [?, 7, 7, 128]
    end_points['inception_3'] = net

    net = inception(net, [64, 48, 64, 64, 96, 32])  # [?, 7, 7, 256]
    end_points['inception_4'] = net

    net = max_pooling_layer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  # [?, 3, 3, 256]
    end_points['pooling_4'] = net

    # Construct Fully Connected Layers
    fc_X = tf.reshape(net, shape=[-1, 3 * 3 * 256])

    fc_L1 = input_layer(fc_X, 3 * 3 * 256, 625, dropout_rate)
    fc_L2 = hidden_layer(fc_L1, 625, 625, dropout_rate)
    fc_L3 = hidden_layer(fc_L2, 625, 625, dropout_rate)
    fc_L4 = hidden_layer(fc_L3, 625, 625, dropout_rate)
    fc_L4S = tf.add(fc_L4, fc_L1)
    fc_L5 = output_layer(fc_L4S, 625, 225, dropout_rate)

    last_fc_L = fc_L5

    return last_fc_L


SAVE_DIR = 'save_files'
import os
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
MODEL_SAVE_PATH_WITHOUT_EXTENSION = "{0}/{1}.{2}".format(SAVE_DIR, os.path.basename(__file__), VERSION)
MODEL_SAVE_PATH = MODEL_SAVE_PATH_WITHOUT_EXTENSION + ".ckpt"

def load_model(sess, saver):
    saver.restore(sess, MODEL_SAVE_PATH)

def remove_model():
    os.remove(MODEL_SAVE_PATH)

def save_model(sess, saver):
    saver.save(sess, MODEL_SAVE_PATH)