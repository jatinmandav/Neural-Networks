import tensorflow as tf
import numpy as np

x = None
y = None

class Trainer:
    def __init__(self, tf_x, tf_y):
        global x, y
        x = tf_x
        y = tf_y

    def train_network(self, x, dataset=[None, None], model=None, learning_rate=0.001, save_model_path="model/model.ckpt", total_epochs=0, batch_size=0):
        [images, labels] = dataset

        test_size = int(0.1*len(images))

        test_img, test_labels = images[:test_size], labels[:test_size]
        train_img, train_labels = images[test_size:], labels[test_size:]

        prediction = model(x)
        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(total_epochs):
                i = 0
                epoch_loss = 0
                while i < len(train_img):
                    start = i
                    end = i + batch_size
                    i = end

                    epoch_img, epoch_label = train_img[start:end], train_labels[start:end]

                    _, c = sess.run([optimizer, cost_func], feed_dict={x: epoch_img, y: epoch_label})
                    epoch_loss += c

                print("Epoch {} Completed of {}. Loss: {}".format(epoch + 1, total_epochs, epoch_loss))
                saver.save(sess, save_model_path)

            print("Accuracy: {}".format(accuracy.eval({x: test_img, y: test_labels})))
