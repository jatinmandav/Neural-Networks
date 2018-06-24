import tensorflow as tf

class Feed_Forward:
    def __init__(self, architecture=None):
        self.neurons = architecture
    def model(self, x):
        # neurons = [784, 100, 400, 100, 10]
        architecture = []
        for i in range(len(self.neurons) - 1):
            layer = {'weights': tf.Variable(tf.random_normal([self.neurons[i], self.neurons[i+1]])),
                    'biases': tf.Variable(tf.random_normal([self.neurons[i+1]]))}

            architecture.append(layer)

        layer_output = x
        for i in range(len(architecture) - 1):
            layer_output = tf.matmul(layer_output, architecture[i]['weights'])
            layer_output = tf.add(layer_output, architecture[i]['biases'])

            layer_output = tf.nn.relu(layer_output)

        layer_output = tf.matmul(layer_output, architecture[-1]['weights'])
        final_output = tf.add(layer_output, architecture[-1]['biases'])

        return final_output
