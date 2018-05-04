"""Class to creAate a small Siamese Neural Network"""
import tensorflow as tf

class SmallSiamese(object):

    def __init__(self):
        self.x1 = tf.palceholder(tf.float32, [None, 299, 299, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 299, 299, 3])

        with tf.variable_scope("siamese") as scope:
            self.embedding1 = self.network(self.x1)
            scope.reuse_variable()
            self.embedding2 = self.network(self.x2)

        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self._get_loss()

    def network(self, x):
        """Define CNN"""

        conv1 = self._get_conv2d(x, self._get_weight_variable([3, 3, 1, 32], "weight1"),
                                 self._get_bias_variable([32], "bias1"),
                                 "conv1")
        pool1 = self._get_maxpool2d(conv1, "pool1")

        conv2 = self._get_conv2d(pool1, self._get_weight_variable([3, 3, 32, 64], "weight2"),
                                 self._get_bias_variable([32], "bias2"),
                                 "conv2")
        pool2 = self._get_maxpool2d(conv2, "pool2")

        conv3 = self._get_conv2d(pool2, self._get_weight_variable([3, 3, 64, 128], "weight3"),
                                 self._get_bias_variable([32], "bias3"),
                                 "conv3")
        pool3 = self._get_maxpool2d(conv3)

        return tf.contrib.layers.flatten(pool3, "pool3")

    def _get_weight_variable(self, shape, name):
        """Generate initialized wieghts"""
        return tf.get_variable(name, shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def _get_bias_variable(self, shape, name):
        """Generate initialized bias"""
        return tf.get_variable(name, shape,
                               initializer=tf.constant_initializer(0.02))

    def _get_conv2d(self, x, W, b, name, strides=1):
        """conv2d wrapper with relu activation"""
        x = tf.nn.conv2d(input=x, filter=W, strides=[1, strides, strides, 1],
                         padding='SAME', name=name)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _get_maxpool2d(self, x, name, k=2):
        """maxpool wrapper"""
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def _get_loss(self, margin=0.2):
        with tf.name_scope("loss"):
            scores = tf.sqrt(tf.reduce_sum(tf.pow(self.embedding1-self.embedding2, 2), 1, keep_dims=True))
            loss = tf.reduce_mean(self.y_*tf.square(scores) + (1-self.y_)*tf.square(tf.maximum((margin-scores), 0)))
            return loss
