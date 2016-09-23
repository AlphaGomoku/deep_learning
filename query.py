import tensorflow as tf
from model.cnn import (make_model, load_model, )


class QueryManager(object):

    def __init__(self):
        self.X = tf.placeholder("float", [None, 225])
        self.dropout_rate = tf.placeholder("float")

        model = make_model(self.X, self.dropout_rate)
        self.model_with_softmax = tf.nn.softmax(model)

        self.sess = tf.Session()

        saver = tf.train.Saver()
        load_model(self.sess, saver)

    def query(self, state):
        result = self.sess.run(tf.argmax(self.model_with_softmax, 1), feed_dict={self.X:[state], self.dropout_rate:1.0})
        return result[0]


if __name__ == '__main__':
    qm = QueryManager()
    res = qm.query([0 for i in range(225)])
    print(res)