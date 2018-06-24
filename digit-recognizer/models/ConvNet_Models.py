import tensorflow as tf

class Multi_Column_DeepNet:
    def __init__(self):
        pass

    def conv_2d(self, x, filter, stride, padding='SAME'):
        return tf.nn.conv2d(x, filter=filter, strides=stride, padding=padding)

    def relu(self, x):
        return tf.nn.relu(x)

    def max_pool_2d(self, x, ksize, stride, padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding=padding)

    def get_weights_biases(self, weight_size, bias_size):
        return tf.Variable(tf.random_normal(weight_size)), tf.Variable(tf.random_normal(bias_size))

    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x, keep_rate)

    def model(self, x):
        # Layer 1: CONVOLUTION -> ReLU -> MAX_POOL
        weights_l1, biases_l1 = self.get_weights_biases([5, 5, 1, 32], [32])

        conv1 = self.conv_2d(x, filter=weights_l1, stride=[1, 1, 1, 1])
        relu1 = self.relu(tf.add(conv1, biases_l1))
        output_layer1 = self.max_pool_2d(relu1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Layer 2: CONVOLUTION -> ReLU -> MAX_POOL
        weights_l2, biases_l2 = self.get_weights_biases([5, 5, 32, 64], [64])

        conv2 = self.conv_2d(output_layer1, filter=weights_l2, stride=[1, 1, 1, 1])
        relu2 = self.relu(tf.add(conv2, biases_l2))
        output_layer2 = self.max_pool_2d(relu2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Flattening 2D Layer
        output_layer2 = tf.reshape(output_layer2, [-1, 7*7*64])

        # Layer 3: FULLY CONNECTED -> ReLU
        weights_l3, biases_l3 = self.get_weights_biases([7*7*64, 1024], [1024])

        output_layer3 = tf.add(tf.matmul(output_layer2, weights_l3), biases_l3)
        output_layer3 = self.relu(output_layer3)

        # DROPOUT LAYER
        dropout_layer = self.dropout(output_layer3, keep_rate=0.8)

        # Layer 4: FULLY CONNECTED -> ReLU
        weights_l4, biases_l4 = self.get_weights_biases([1024, 10], [10])

        output_layer4 = tf.add(tf.matmul(dropout_layer, weights_l4), biases_l4)

        return output_layer4


class Vote_Multi_Column_DeepNet(Multi_Column_DeepNet):
    def __init__(self):
        pass

    def network(self, x):
        # Layer 1: CONVOLUTION -> ReLU -> MAX_POOL
        weights_l1, biases_l1 = self.get_weights_biases([5, 5, 1, 32], [32])

        conv1 = self.conv_2d(x, filter=weights_l1, stride=[1, 1, 1, 1])
        relu1 = self.relu(tf.add(conv1, biases_l1))
        output_layer1 = self.max_pool_2d(relu1, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Layer 2: CONVOLUTION -> ReLU -> MAX_POOL
        weights_l2, biases_l2 = self.get_weights_biases([5, 5, 32, 64], [64])

        conv2 = self.conv_2d(output_layer1, filter=weights_l2, stride=[1, 1, 1, 1])
        relu2 = self.relu(tf.add(conv2, biases_l2))
        output_layer2 = self.max_pool_2d(relu2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Flattening 2D Layer
        output_layer2 = tf.reshape(output_layer2, [-1, 7*7*64])

        # Layer 3: FULLY CONNECTED -> ReLU
        weights_l3, biases_l3 = self.get_weights_biases([7*7*64, 1024], [1024])

        output_layer3 = tf.add(tf.matmul(output_layer2, weights_l3), biases_l3)
        output_layer3 = self.relu(output_layer3)

        # DROPOUT LAYER
        dropout_layer3 = self.dropout(output_layer3, keep_rate=0.8)

        # Layer 4: FULLY CONNECTED -> ReLU
        weights_l4, biases_l4 = self.get_weights_biases([1024, 1024], [1024])

        output_layer4 = tf.add(tf.matmul(dropout_layer3, weights_l4), biases_l4)
        output_layer4 = self.relu(output_layer4)

        # DROPOUT LAYER
        dropout_layer4 = self.dropout(output_layer4, keep_rate=0.8)

        return dropout_layer4

    def model(self, x):
        series_1 = self.network(x)
        series_2 = self.network(x)
        series_3 = self.network(x)

        concated_layer = tf.concat([series_1, series_2, series_3], 1)

        weights, biases = self.get_weights_biases([1024*3, 10], [10])

        output_layer = tf.add(tf.matmul(concated_layer, weights), biases)

        return output_layer










































#
