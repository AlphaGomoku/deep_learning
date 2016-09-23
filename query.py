import tensorflow as tf
from model.cnn import (make_model, load_model, )

class QueryManager(object):

    def __init__(self):
        self.X = tf.placeholder("float", [None, 225])
        self.dropout_rate = tf.placeholder("float")

        self.model = make_model(self.X, self.dropout_rate)
        self.model_with_softmax = tf.nn.softmax(self.model)

        self.sess = tf.Session()

        saver = tf.train.Saver()
        load_model(self.sess, saver)

    def query(self, state):

        res = self.sess.run(self.model, feed_dict={self.X:[state], self.dropout_rate:1.0})[0]

        best_idx = 0
        best_val = res[0]
        for idx, val in enumerate(res):
            if state[idx] == 0 and best_val < val:
                best_idx = idx
                best_val = val

        return best_idx


if __name__ == '__main__':
    qm = QueryManager()
    res = qm.query([0 for i in range(225)])
    print(res)