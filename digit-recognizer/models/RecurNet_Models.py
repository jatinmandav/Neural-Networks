import tensorflow as tf
from tensorflow.contrib import rnn

class Simple_RNN:
    def __init__(self, hidden_features=0, no_classes=0, timesteps=0):
        self.hidden_features = hidden_features
        self.no_classes = no_classes
        self.timesteps = timesteps

    def model(self, x):
        weights = tf.Variable(tf.random_normal([self.hidden_features, self.no_classes]))
        biases = tf.Variable(tf.random_normal([self.no_classes]))

        x = tf.unstack(x, self.timesteps, 1)

        lstm_instance = rnn.BasicLSTMCell(self.hidden_features)
        rnn_output, states = rnn.static_rnn(lstm_instance, x, dtype=tf.float32)

        output = tf.add(tf.matmul(rnn_output[-1], weights), biases)
        return output

class Bidirectional_RNN:
    def __init__(self, hidden_features=0, no_classes=0, timesteps=0):
        self.hidden_features = hidden_features
        self.no_classes = no_classes
        self.timesteps = timesteps

    def model(self, x):
        weights = tf.Variable(tf.random_normal([self.hidden_features*2, self.no_classes]))
        biases = tf.Variable(tf.random_normal([self.no_classes]))

        x = tf.unstack(x, self.timesteps, 1)

        lstm_forward = rnn.BasicLSTMCell(self.hidden_features)
        lstm_backward = rnn.BasicLSTMCell(self.hidden_features)

        rnn_output, _, _ = rnn.static_bidirectional_rnn(lstm_forward, lstm_backward, x, dtype=tf.float32)

        output = tf.add(tf.matmul(rnn_output[-1], weights), biases)
        return output
