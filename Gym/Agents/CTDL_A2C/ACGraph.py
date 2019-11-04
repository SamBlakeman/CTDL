import os
import tensorflow as tf
import numpy as np


class ACGraph(object):
    def __init__(self, input_dim, action_mins, action_maxs, directory):
        self.ti = 0
        self.input_dim = input_dim
        self.action_mins = action_mins
        self.action_maxs = action_maxs
        self.action_dim = action_mins.shape[0]
        self.directory = directory

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            """ Construction phase """
            self.init_xavier = tf.contrib.layers.xavier_initializer()

            self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="X")

            self.action_y = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="action_y")
            self.value_y = tf.placeholder(tf.float32, shape=(None), name="value_y")
            self.delta = tf.placeholder(tf.float32, shape=(None), name="delta")

            # Layers
            self.dense1_val = tf.layers.dense(inputs=self.X, units=128, activation=tf.nn.elu, kernel_initializer=self.init_xavier)
            self.dense2_val = tf.layers.dense(inputs=self.dense1_val, units=128, activation=tf.nn.elu, kernel_initializer=self.init_xavier)
            self.state_value = tf.layers.dense(inputs=self.dense2_val, units=1, activation=None, kernel_initializer=self.init_xavier)

            self.dense1_pol = tf.layers.dense(inputs=self.X, units=128, activation=tf.nn.elu, kernel_initializer=self.init_xavier)
            self.dense2_pol = tf.layers.dense(inputs=self.dense1_pol, units=128, activation=tf.nn.elu, kernel_initializer=self.init_xavier)
            self.action_means = tf.layers.dense(inputs=self.dense2_pol, units=self.action_dim, activation=None, kernel_initializer=self.init_xavier)
            self.action_sigmas = tf.nn.softplus(tf.layers.dense(inputs=self.dense2_pol, units=self.action_dim, activation=None, kernel_initializer=self.init_xavier))

            self.dist = tf.contrib.distributions.Normal(loc=self.action_means, scale=self.action_sigmas)
            self.action_sample = tf.squeeze(self.dist.sample(1), axis=0)
            self.action = tf.clip_by_value(self.action_sample, self.action_mins[0], self.action_maxs[0])

            # Loss functions
            with tf.name_scope("loss"):
                self.policy_loss = (-tf.log(self.dist.prob(self.action_y) + 1e-5) * self.delta)# - self.entropy
                self.value_loss = tf.reduce_mean(tf.square(self.value_y - self.state_value), axis=0, name='value_loss')

            # Minimizer
            self.learning_rate_policy = 0.00001
            self.learning_rate_value = 0.0001

            with tf.name_scope("train"):
                self.training_op_policy = tf.train.AdamOptimizer(self.learning_rate_policy, name='optimizer').minimize(self.policy_loss)
                self.training_op_value = tf.train.AdamOptimizer(self.learning_rate_value, name='optimizer').minimize(self.value_loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            tf.add_to_collection('action', self.action)
            tf.add_to_collection('state_value', self.state_value)

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        return

    def GetAction(self, X):

        action = self.action.eval(feed_dict={self.X: X}, session=self.sess)

        return action

    def GetStateValue(self, X):

        value = self.state_value.eval(feed_dict={self.X: X}, session=self.sess)

        return value

    def GradientDescentStep(self, X_batch, action_batch, value_batch, delta_batch):

        self.sess.run(self.training_op_policy, feed_dict={self.X: X_batch,
                                                          self.action_y: action_batch,
                                                          self.delta: np.squeeze(delta_batch)})

        self.sess.run(self.training_op_value, feed_dict={self.X: X_batch, self.value_y: np.squeeze(value_batch)})

        return

    def SaveGraphAndVariables(self):
        save_path = self.saver.save(self.sess, self.directory)
        print('Model saved in ' + save_path)

        return

    def LoadGraphAndVariables(self):
        self.saver.restore(self.sess, self.directory)
        print('Model loaded from ' + self.directory)

        return






