"""Class to creAate a small Siamese Neural Network"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from landmark.utils.configurable import Configurable
from landmark.domain.preprocessing.preprocessing_landmark_recognition import LandmarkRecognitionAlbum

class SmallSiamese(Configurable):

    LOG_ENV = "siamese/small_siamese/logs"

    def __init__(self, path_to_config, init_file):
        super(SmallSiamese, self).__init__(path_to_config)
        self.init_file = init_file
        self.log_name = self.init_file["log_name"]
        self.nb_data_train = self.init_file["nb_data_train"]
        self.batch_size = self.init_file["batch_size"]
        self.init_learning_rate = self.init_file["init_learning_rate"]
        self.decay_rate = self.init_file["decay_rate"]
        
        self.cls = self.__class__
        self.warehouse = self.config["data"]["warehouse"]
        self.path_to_logs = os.path.join(self.warehouse, self.cls.LOG_ENV, self.log_name)
        self.sess = tf.Session()
        tf.set_random_seed(42)

        self.x1 = tf.placeholder(tf.float32, [None, 299, 299, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 299, 299, 3])
        self.y_ = tf.placeholder(tf.float32, [None, 2], name="y_true")

        with tf.variable_scope("siamese") as scope:
            self.embedding1 = self._network(self.x1)
            tf.get_variable_scope().reuse_variables()
            self.embedding2 = self._network(self.x2)
            self.proximity = self.embedding1-self.embedding2

            self.fc = self._get_fully_connected(self.proximity,
                                                self._get_weight_variable([self.proximity.shape[1], 2], "weight_fc"),
                                                self._get_bias_variable([2], "bias_fc"),
                                                "fully_connected")

            with tf.name_scope("softmax"):
                self.y_pred = tf.nn.softmax(self.fc)

            with tf.name_scope("accuracy_val"):
                self.accuracy, self.accuracy_update = tf.metrics.accuracy(labels=tf.argmax(self.y_, 1),
                                                                          predictions=tf.argmax(self.y_pred, 1),
                                                                          name="accuracy_validation")
            # Cross entropy loss
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred,
                                                                            labels=self.y_)
            with tf.name_scope("cross_entropy"):
                self.loss = tf.reduce_mean(self.cross_entropy)

            # Define optimizer with decreasing learning rate
            self.batch_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning_rate,
                                                            global_step=self.batch_step*self.batch_size,
                                                            decay_steps=self.nb_data_train,
                                                            decay_rate=self.decay_rate,
                                                            staircase=True)

            with tf.variable_scope("train", reuse=tf.AUTO_REUSE): 
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Create summary for loss
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy_val", self.accuracy)
        self.summary_op = tf.summary.merge_all()

        # Writer object for tensor logs
        self.writer_train = tf.summary.FileWriter(self.path_to_logs, graph=tf.get_default_graph())

        # Need to reset values of accuracy for validation / test
        self.stream_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_val")
        self.reset_ops_accuracy = tf.variables_initializer(self.stream_vars)

        # Initialize variables
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_batch, iteration, nb_iter):
        """
        Method to train the small siamese network

        PARAMETERS
        ----------
        train_batch: Batch object
            object generating training batch (labels & data) at each iteration
        iteration: int
            value of the current iteration
        nb_iter: int
            value of the total number of iteration to do

        RETURNS
        -------
        loss: float
            loss value for a given training batch during the current iteration
        """

        X_train_batch = train_batch.batch_data
        y_train_batch = train_batch.batch_label
        batch_imgs, exceptions = LandmarkRecognitionAlbum(X_train_batch).load
        batch_x1 = self._structure_data(batch_imgs, 0)
        batch_x2 = self._structure_data(batch_imgs, 1)
        batch_labels = self._structure_labels(y_train_batch)
        batch_labels = np.delete(batch_labels, exceptions, axis=0) # Delete labels of unloaded images

        _, loss, summary = self.sess.run([self.optimizer, self.loss, self.summary_op],
                                         feed_dict={self.x1: batch_x1,
                                                    self.x2: batch_x2,
                                                    self.y_: batch_labels,
                                                    self.batch_step: iteration})
        self.writer_train.add_summary(summary, iteration)
        print("***** Training Loss at step %s/%s: %s *****" % (iteration, nb_iter, str(loss)))
        print("**********")
        train_batch.next
        return loss

    def validation(self, val_batch, iteration, nb_iter):
        accuracies = list()
        nb_batch = int(np.ceil(val_batch.nb_data/val_batch.batch_size))

        for i in range(nb_batch):
            X_val_batch = val_batch.batch_data
            y_val_batch = val_batch.batch_label
            batch_imgs, exceptions = LandmarkRecognitionAlbum(X_val_batch).load
            batch_x1 = self._structure_data(batch_imgs, 0)
            batch_x2 = self._structure_data(batch_imgs, 1)
            batch_labels = self._structure_labels(y_val_batch)
            batch_labels = np.delete(batch_labels, exceptions, axis=0) # Delete labels of unloaded images

            self.sess.run(self.reset_ops_accuracy)
            self.sess.run([self.accuracy_update],
                          feed_dict={self.x1: batch_x1,
                                     self.x2: batch_x2,
                                     self.y_: batch_labels})
            current_accuracy, summary = self.sess.run([self.accuracy, self.summary_op],
                                                      feed_dict={self.x1: batch_x1,
                                                                 self.x2: batch_x2,
                                                                 self.y_: batch_labels})
            self.writer_train.add_summary(summary, iteration)
            accuracies.append(current_accuracy)

        accuracy = np.mean(accuracies)
        print("***** Validation (Mean) Accuracy at Iteration %s/%s: %s" % (iteration, nb_iter, str(accuracy)))
        print("***********")
        return accuracy

    def _network(self, x):
        """Define CNN architecture"""

        conv1 = self._get_conv2d(x, self._get_weight_variable([3, 3, 3, 32], "weight1"),
                                 self._get_bias_variable([32], "bias1"),
                                 "conv1")
        pool1 = self._get_maxpool2d(conv1, "pool1")

        conv2 = self._get_conv2d(pool1, self._get_weight_variable([3, 3, 32, 64], "weight2"),
                                 self._get_bias_variable([64], "bias2"),
                                 "conv2")
        pool2 = self._get_maxpool2d(conv2, "pool2")

        conv3 = self._get_conv2d(pool2, self._get_weight_variable([3, 3, 64, 128], "weight3"),
                                 self._get_bias_variable([128], "bias3"),
                                 "conv3")
        pool3 = self._get_maxpool2d(conv3, "pool3")
        
        return tf.contrib.layers.flatten(pool3, "embedding")

    def _get_weight_variable(self, shape, name):
        """Generate initialized wieghts"""
        with tf.variable_scope("name", reuse=tf.AUTO_REUSE):
            return tf.get_variable(name, shape,
                                   initializer=tf.contrib.layers.xavier_initializer())

    def _get_bias_variable(self, shape, name):
        """Generate initialized bias"""
        with tf.variable_scope("name", reuse=tf.AUTO_REUSE):
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

    def _get_fully_connected(self, x, W, b, name):
        """wrapper for fully connected with relu activation"""
        with tf.variable_scope(name):
            layer = tf.matmul(x, W) + b
            layer = tf.nn.relu(layer)
            return layer

    def _structure_data(self, batch_imgs, index_batch):
        """Method to structure images to the exact format"""
        batch = [batch_im[index_batch] for batch_im in batch_imgs]
        batch = np.expand_dims(batch, axis=0)[0]
        return batch

    def _structure_labels(self, data):
        """Method to structure labels to the exact format"""
        labels = pd.DataFrame(data, columns=["similarity"])
        labels["similarity_env"] = 1 - labels["similarity"]
        labels = labels.as_matrix()
        return labels
