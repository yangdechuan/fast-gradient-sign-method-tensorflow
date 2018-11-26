import os
import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class RobustMLP(object):
    def __init__(self, epsilon=0.1):
        """Robust MLP Model."""
        self.in_units = 784
        self.h1_units = 300
        self.num_classes = 10

        self.lr = 0.1  # learning rate
        self.epsilon = epsilon  # perturb rate

        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        logging.basicConfig(level=logging.INFO)
    
    def load_data(self, dataset="mnist"):
        """Load train and test dataset.

        Arguments:
            dataset: 'mnist' or 'fashioin-mnist'
        
        Load (self.x_train, self.y_train), (self.x_test, self.y_test)
            x_train: shape=[60000, 784], dtype=float64
            y_train: shape=[60000, 10], dtype=float32
            x_test: shape=[10000, 784], dtype=float64
            y_test: shape=[10000, 10], dtype=float32
        """
        logging.info("Loading dataset...")
        if dataset == "mnist":
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        elif dataset == "fashion-mnist":
            (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28) # reshape
        x_train, x_test = x_train / 255.0, x_test / 255.0 # normalization
        y_train = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        logging.info("Load data success.")
    
    def basic_model(self):
        """Test basic mlp accuracy on adversarial samples."""
        def mlp(x, reuse=None):
            """Define a mlp function."""
            with tf.variable_scope("fc", reuse=reuse):
                W1 = tf.get_variable("W1", shape=[self.in_units, self.h1_units], dtype=tf.float32)
                b1 = tf.get_variable("b1", shape=[self.h1_units], dtype=tf.float32)
                W2 = tf.get_variable("W2", shape=[self.h1_units, self.num_classes], dtype=tf.float32)
                b2 = tf.get_variable("b2", shape=[self.num_classes], dtype=tf.float32)
                tf.summary.histogram("W1", W1)
                tf.summary.histogram("b1", b1)
                tf.summary.histogram("W2", W2)
                tf.summary.histogram("b2", b2)
                fc1 = tf.matmul(x, W1) + b1
                relu = tf.nn.relu(fc1)
                fc2 = tf.matmul(relu, W2) + b2
                y = tf.nn.softmax(fc2)
                return y
        x = tf.placeholder(tf.float32, shape=[None, self.in_units], name="x")  # network input
        y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="y_true")  # gold label
        y = mlp(x)  # network output

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        # train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)  # not convergence
        train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for i in range(3000):
            # train model
            s = (i * 100) % 60000
            t = s + 100
            batch_xs = self.x_train[s:t]
            batch_ys = self.y_train[s:t]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            acc = sess.run(accuracy, feed_dict={x: self.x_test, y_: self.y_test})
            print("accuracy on test dta: {}".format(acc))
        
        # test on adversarial samples
        perb = self.epsilon * tf.sign(tf.gradients(cross_entropy, x)[0])  # calculate perturbation
        perb = sess.run(perb, feed_dict={x: self.x_test, y_: self.y_test})
        x_perb = self.x_test + perb
        # clip
        x_perb[x_perb > 1.0] = 1.0
        x_perb[x_perb < 0.0] = 0.0
        acc = sess.run(accuracy, feed_dict={x: x_perb, y_: self.y_test})
        print("accuracy on adversarial samples: {}".format(acc))
    
    def robust_model(self):
        """Train robust mlp and test accuracy on adversarial samples."""
        def mlp(x, reuse=None):
            """Define a mlp function."""
            with tf.variable_scope("fc", reuse=reuse):
                W1 = tf.get_variable("W1", shape=[self.in_units, self.h1_units], dtype=tf.float32)
                b1 = tf.get_variable("b1", shape=[self.h1_units], dtype=tf.float32)
                W2 = tf.get_variable("W2", shape=[self.h1_units, self.num_classes], dtype=tf.float32)
                b2 = tf.get_variable("b2", shape=[self.num_classes], dtype=tf.float32)
                tf.summary.histogram("W1", W1)
                tf.summary.histogram("b1", b1)
                tf.summary.histogram("W2", W2)
                tf.summary.histogram("b2", b2)
                fc1 = tf.matmul(x, W1) + b1
                relu = tf.nn.relu(fc1)
                fc2 = tf.matmul(relu, W2) + b2
                y = tf.nn.softmax(fc2)
                return y

        x = tf.placeholder(tf.float32, shape=[None, self.in_units], name="x")  # network input
        y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="y_true")  # gold label
        y = mlp(x)  # network output
        # Loss and optimizer.
        # Loss is defined as: 0.5 * J(w, x, y) + 0.5 * J(w, x + epsilon * sign(Grad_x_J(w, x, y)), y)
        cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        perb = self.epsilon * tf.sign(tf.gradients(cross_entropy1, x)[0])  # calculate perturbation
        x_perb = x + perb  # adversarial samples
        y_perb = mlp(x_perb, reuse=True)  # network output in adversarial samples
        cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_perb), axis=[1]))
        cross_entropy = 0.5 * cross_entropy1 + 0.5 * cross_entropy2

        optimizer = tf.train.AdagradOptimizer(self.lr)
        train_step = optimizer.minimize(cross_entropy)

        # Define accuracy.
        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        # Train and test model.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summ_fc = tf.summary.merge_all(scope="fc")
        summ_acc = tf.summary.merge_all(scope="metrics")

        writer = tf.summary.FileWriter(os.path.join("tmp", "mnist"))
        writer.add_graph(sess.graph)

        for i in range(3000):
            # train model
            s = (i * 100) % 60000
            t = s + 100
            batch_xs = self.x_train[s:t]
            batch_ys = self.y_train[s:t]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # accuracy and loss
            [acc] = sess.run([accuracy], feed_dict={x: self.x_test, y_: self.y_test})
            print("Test accuracy: {}".format(acc))
            # summary
            [summ_fc_tmp] = sess.run([summ_fc], feed_dict={x: batch_xs, y_: batch_ys})
            [summ_acc_tmp] = sess.run([summ_acc], feed_dict={x: self.x_test, y_: self.y_test})
            writer.add_summary(summ_fc_tmp, global_step=i)
            writer.add_summary(summ_acc_tmp, global_step=i)
        
        # test on adversarial samples
        perb = sess.run(perb, feed_dict={x: self.x_test, y_: self.y_test})
        x_perb = self.x_test + perb
        # clip
        x_perb[x_perb > 1.0] = 1.0
        x_perb[x_perb < 0.0] = 0.0

        # show
        for i in range(1, 17):
            plt.subplot(4, 4, i)
            plt.imshow(self.x_test[i].reshape([28, 28]) * 255.0)
        plt.savefig("original_samples.jpg")
        plt.clf()
        for i in range(1, 17):
            plt.subplot(4, 4, i)
            plt.imshow(x_perb[i].reshape([28, 28]) * 255.0)
        plt.savefig("perturbed_samples.jpg")
        plt.clf()

        acc = sess.run(accuracy, feed_dict={x: x_perb, y_: self.y_test})
        print("accuracy on adversarial samples: {}".format(acc))


if __name__ == "__main__":
    # epsilon=0.1 or epsilon=0.25
    mlp = RobustMLP(epsilon=0.1)
    # dataset='mnist' or dataset='fashion-mnist'
    mlp.load_data(dataset="mnist")

    # two model to test
    mlp.basic_model()
    #mlp.robust_model()

