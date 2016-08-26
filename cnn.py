import tensorflow as tf
import numpy as np
import io_data


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


train_x_data, train_y_data, test_x_data, test_y_data = io_data.get_train_test_data(one_hot = True)
train_data_len = len(train_x_data)
test_data_len = len(test_x_data)

X = tf.placeholder("float", [None, 225])
Y = tf.placeholder("float", [None, 225])

dropout_rate = tf.placeholder("float")

model = make_model(X, dropout_rate)
model_with_softmax = tf.nn.softmax(model)

# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

LEARNING_RATE = 0.001
learning_rate = tf.Variable(LEARNING_RATE)
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    """
        Variables and functions about
        Loading and Saving Data.
    """
    saver = tf.train.Saver()
    SAVE_DIR = 'save_files'
    import os
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    MODEL_SAVE_PATH = "{0}/{1}.{2}.ckpt".format(SAVE_DIR, os.path.basename(__file__), VERSION)
    INFO_FILE_PATH = "{0}/{1}.{2}.info".format(SAVE_DIR, os.path.basename(__file__), VERSION)

    def do_load():
        start_epoch = 1
        try:
            epochs = []
            avg_costs = []
            avg_accuracys = []
            learning_rates = []

            with open(INFO_FILE_PATH, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    data = line.split()
                    epochs.append(int(data[0]))
                    avg_costs.append(float(data[1]))
                    avg_accuracys.append(float(data[2]))
                    learning_rates.append(float(data[3]))
            saver.restore(sess, MODEL_SAVE_PATH)
            print("[*] The save file exists!")

            print("Do you wanna continue? (y/n) ", end="", flush=True)
            if input() == 'n':
                print("not continue...")
                print("[*] Start a training from the beginning.")
                os.remove(INFO_FILE_PATH)
                os.remove(MODEL_SAVE_PATH)
                sess.run(init)
            else:
                print("continue...")
                print("[*] Start a training from the save file.")
                start_epoch = epochs[-1] + 1
                for epoch, avg_cost, avg_accuracy, learning_rate in zip(epochs, avg_costs, avg_accuracys,
                                                                        learning_rates):
                    print("Epoch {0} with learning rate = {1} : avg_cost = {2}, avg_accuracy = {3}".
                          format(epoch, learning_rate, avg_cost, avg_accuracy))

        except FileNotFoundError:
            print("[*] There is no save files.")
            print("[*] Start a training from the beginning.")
            sess.run(init)

        return start_epoch

    def do_save():
        print("[progress] Saving result! \"Never\" exit!!", end='', flush=True)
        saver.save(sess, MODEL_SAVE_PATH)
        with open(INFO_FILE_PATH, "a") as f:
            f.write("{0} {1} {2} {3}".format(epoch, avg_cost, avg_accuracy, LEARNING_RATE) + os.linesep)
        print("", end='\r', flush=True)


    """
        Variables and functions about
        Training and Testing Model
    """
    DISPLAY_SAVE_STEP = 1
    TRAINING_EPOCHS = 1000
    BATCH_SIZE = 2048

    def do_train():
        print("[progress] Training model for optimizing cost!", end='', flush=True)
        # Loop all batches for training
        avg_cost = 0
        for start in range(0, train_data_len, BATCH_SIZE):
            end = min(start + BATCH_SIZE, train_data_len)
            batch_x = train_x_data[start:end]
            batch_y = train_y_data[start:end]
            data = {X: batch_x, Y: batch_y, dropout_rate: 0.5}
            sess.run(train, feed_dict=data)
            avg_cost += sess.run(cost, feed_dict=data) * len(batch_x) / train_data_len

        print("", end='\r', flush=True)
        return avg_cost

    def do_test():
        print("[progress] Testing model for evaluating accuracy!", end='', flush=True)
        correct_prediction = tf.equal(tf.argmax(model_with_softmax, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Loop all batches for test
        avg_accuracy = 0
        for start in range(0, test_data_len, BATCH_SIZE):
            end = min(start + BATCH_SIZE, test_data_len)
            batch_x = test_x_data[start:end]
            batch_y = test_y_data[start:end]
            data = {X: batch_x, Y: batch_y, dropout_rate: 1.0}
            avg_accuracy += accuracy.eval(data) * len(batch_x) / test_data_len

        print("", end='\r', flush=True)
        return avg_accuracy


    ################################## Start of flow ##################################

    start_epoch = do_load()

    if start_epoch == 1:
        avg_accuracy = do_test()
        print("After initializing, accuracy = {0}".format(avg_accuracy))

    # Training cycle
    for epoch in range(start_epoch, TRAINING_EPOCHS + 1):

        avg_cost = do_train()

        # Logging the result
        if epoch % DISPLAY_SAVE_STEP == start_epoch % DISPLAY_SAVE_STEP or epoch == TRAINING_EPOCHS:
            avg_accuracy = do_test()
            do_save()

            # Print Result
            print("Epoch {0} : avg_cost = {1}, accuracy = {2}".format(epoch, avg_cost, avg_accuracy))
