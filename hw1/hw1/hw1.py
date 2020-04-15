import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = labels[:, -1, :, :],
        logits = preds[:, -1, :, :]
        ))

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """

        K = self.samples_per_class - 1
        N = self.num_classes
        I = 784

        # concat input seq should be of size [B, K*N+N, 784 + 10]
        # where the last N are the ones we wish to predict,
        # and the 10 features are the label

        train_images, test_images = tf.split(input_images, [K, 1], 1)
        train_labels, test_labels = tf.split(input_labels, [K, 1], 1)

        test_images = tf.reshape(test_images, [tf.shape(test_images)[0], N, I])
        train_images = tf.reshape(train_images, [tf.shape(train_images)[0], K*N, I])

        test_labels = tf.zeros([tf.shape(test_labels)[0], N, N])
        train_labels = tf.reshape(train_labels, [tf.shape(train_labels)[0], K*N, N])

        train = tf.concat([train_images, train_labels], -1)
        test = tf.concat([test_images, test_labels], -1)

        concat_input_seq = tf.concat([train, test], 1)
        layer1_out = self.layer1(concat_input_seq)
        layer2_out = self.layer2(layer1_out)

        out = tf.reshape(layer2_out, [tf.shape(layer2_out)[0], K+1, N, N])

        return out

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
