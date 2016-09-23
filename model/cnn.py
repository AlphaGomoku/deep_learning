import tensorflow as tf


VERSION = "v.beta"


initializer = tf.contrib.layers.xavier_initializer()

def input_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt = 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))


def hidden_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.nn.relu(tf.add(tf.matmul(X, W), b))


def output_layer(X, num_input, num_output, dropout_rate = None):
    global fc_layer_cnt
    fc_layer_cnt += 1
    if dropout_rate is not None:
        X = tf.nn.dropout(X, dropout_rate)
    W = tf.get_variable("FC_W" + str(fc_layer_cnt), shape=[num_input, num_output], initializer=initializer)
    b = tf.Variable(name="FC_b" + str(fc_layer_cnt), initial_value=tf.random_uniform([num_output], -1.0, 1.0))
    return tf.add(tf.matmul(X, W), b)


def conv_layer(X, shape, strides = [1, 1, 1, 1]):
    global conv_layer_cnt
    conv_layer_cnt += 1
    W = tf.get_variable("Conv_W" + str(conv_layer_cnt), shape = shape, initializer=initializer)
    return tf.nn.conv2d(X, W, strides=strides, padding="SAME")


def relu_layer(X):
    return tf.nn.relu(X)


def pooling_layer(X, ksize, strides):
    return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding="SAME")


def make_model(X, dropout_rate):
    # X's shape is [?, 225]

    # Construct Conv / ReLU / Pooling layers
    conv_X = tf.reshape(X, shape=[-1, 15, 15, 1])   # [?, 15, 15, 1]

    global conv_layer_cnt
    conv_layer_cnt = 0

    L1 = conv_layer(conv_X, shape=[3, 3, 1, 32], strides=[1, 1, 1, 1])  # [?, 15, 15, 32]
    L1R = relu_layer(L1)
    L2 = conv_layer(L1R, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1])  # [?, 15, 15, 32]
    L2R = relu_layer(L2)
    L3 = conv_layer(L2R, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1])  # [?, 15, 15, 32]
    L3S = tf.add(L3, L1)
    L3R = relu_layer(L3S)

    L3P = pooling_layer(L3R, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])  # [?, 8, 8, 32]

    L4 = conv_layer(L3P, shape=[3, 3, 32, 64], strides=[1, 1, 1, 1])  # [?, 8, 8, 64]
    L4R = relu_layer(L4)
    L5 = conv_layer(L4R, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1])  # [?, 8, 8, 64]
    L5R = relu_layer(L5)
    L6 = conv_layer(L5R, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1])  # [?, 8, 8, 64]
    L6S = tf.add(L6, L4)
    L6R = relu_layer(L6S)

    L6P = pooling_layer(L6R, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])  # [?, 4, 4, 64]

    """
    L7 = conv_layer(L6P, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1])   # [?, 4, 4, 128]
    L7R = relu_layer(L7)
    L8 = conv_layer(L7R, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1])  # [?, 4, 4, 128]
    L8R = relu_layer(L8)
    L9 = conv_layer(L8R, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1])  # [?, 4, 4, 128]
    L9S = tf.add(L9, L7)
    L9R = relu_layer(L9)

    L9P = pooling_layer(L9R, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])  # [?, 2, 2, 128]
    """

    last_conv_L = L6P

    # Construct Fully Connected Layers
    fc_X = tf.reshape(last_conv_L, shape=[-1, 4 * 4 * 64])

    fc_L1 = input_layer(fc_X, 4 * 4 * 64, 625, dropout_rate)
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
print(MODEL_SAVE_PATH_WITHOUT_EXTENSION)
MODEL_SAVE_PATH = MODEL_SAVE_PATH_WITHOUT_EXTENSION + ".ckpt"

def load_model(sess, saver):
    saver.restore(sess, MODEL_SAVE_PATH)

def remove_model():
    os.remove(MODEL_SAVE_PATH)

def save_model(sess, saver):
    saver.save(sess, MODEL_SAVE_PATH)
