import os
import tensorflow as tf
import numpy as np


class QGraph(object):
    def __init__(self, num_actions, directory, maze_size):
        self.ti = 0
        self.num_actions = num_actions
        self.directory = directory
        self.maze_size = maze_size

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            """ Construction phase """
            self.X = tf.placeholder(tf.float32, shape=(None, 2), name="X")
            self.y = tf.placeholder(tf.float32, shape=(None), name="y")
            self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions], name="actions")

            # Layers
            self.dense1 = tf.layers.dense(inputs=self.X, units=128, activation=tf.nn.relu)
            self.dense2 = tf.layers.dense(inputs=self.dense1, units=128, activation=tf.nn.relu)
            self.logits = tf.layers.dense(inputs=self.dense2, units=self.num_actions)

            # Loss function
            with tf.name_scope("loss"):
                self.predictions = tf.reduce_sum(tf.multiply(self.logits, self.actions), 1)
                self.targets = tf.stop_gradient(self.y)

                self.error = self.targets - self.predictions
                self.clipped_error = tf.clip_by_value(self.targets - self.predictions, -1., 1.)
                self.loss = tf.reduce_mean(tf.multiply(self.error, self.clipped_error), axis=0, name='loss')

            # Minimizer
            self.learning_rate = 0.00025
            self.momentum = 0.95
            self.epsilon = 0.01

            with tf.name_scope("train"):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, epsilon=self.epsilon)
                self.training_op = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            tf.add_to_collection('logits', self.logits)

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        return

    def GetActionValues(self, X):
        preds = self.logits.eval(feed_dict={self.X: X / self.maze_size}, session=self.sess)

        return preds

    def GradientDescentStep(self, X_batch, action_batch, y_batch):
        actions = np.zeros((X_batch.shape[0], self.num_actions))
        for i in range(X_batch.shape[0]):
            actions[i, action_batch[i]] = 1

        self.sess.run(self.training_op,
                      feed_dict={self.X: X_batch / self.maze_size, self.y: y_batch, self.actions: actions})

        return

    def SaveGraphAndVariables(self):
        save_path = self.saver.save(self.sess, self.directory)
        print('Model saved in ' + save_path)

        return

    def LoadGraphAndVariables(self):
        self.saver.restore(self.sess, self.directory)
        print('Model loaded from ' + self.directory)

        return






