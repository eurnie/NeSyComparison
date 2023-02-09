from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import numpy as np
import tensorflow as tf

sys.path.insert(1, '../data')
from import_data_python_2 import addition_with_only_one_x_value

####################
# hyperparameters
####################

####################
# setup
####################

####################
# training and testing
####################

FLAGS = None

class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      # images = images.astype(numpy.float32)
      # images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def image_processing(x_images):
  # with tf.name_scope('gaussian_noise'):
  #   noise = tf.random_normal(tf.shape(x_images), mean = 0.0, stddev = 0.3, dtype=tf.float32)
  #   x_images = x_images + noise
  # with tf.name_scope('crop'):
  #   x_images = tf.reshape(x_images, [-1,28,28,1])
  #   x_images = tf.random_crop(x_images, [FLAGS.batch_size, 25, 25, 1])
  #   x_images = tf.image.resize_image_with_crop_or_pad(x_images, 28, 28)
  #   x_images = tf.reshape(x_images, [-1,784])
  return x_images

def deepnn(x_images):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  keep_prob = tf.placeholder(tf.float32)

  # x_images = standarization(x_images)
  # x_images = tf.cond(keep_prob <= 1.0, lambda: image_processing(x_images), lambda: x_images)
 
  with tf.name_scope('fc1'):
    h_fc1 = tf.contrib.layers.fully_connected(x_images, 512)
    
  with tf.name_scope('fc2'):
    h_fc2 = tf.contrib.layers.fully_connected(h_fc1, 512)
    
  with tf.name_scope('fc3'):
    h_fc3 = tf.contrib.layers.fully_connected(h_fc2, 256)
    
  with tf.name_scope('fc4'):
    h_fc4 = tf.contrib.layers.fully_connected(h_fc3, 256)

  with tf.name_scope('fc5'):
    h_fc5 = tf.layers.batch_normalization(tf.contrib.layers.fully_connected(h_fc4, 256))
  
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

  # Map the 1024 features to 19 classes, one for each possible outcome
  with tf.name_scope('fc6'):
    y_mlp = tf.contrib.layers.fully_connected(h_fc5_drop, 19, activation_fn = None)

  return y_mlp, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def leakyRelu(value, alpha=0.01):
  return tf.maximum(value, alpha*value)

def main(_):
  # Import data
  # mnist = read_data_sets(FLAGS.data_path, n_labeled=FLAGS.num_labeled, one_hot=True)

  training_set = addition_with_only_one_x_value(1, "train")
  
  train_images = []
  train_labels = []

  for index in range(len(training_set)):
    train_images.append(training_set[index][0][0].numpy())
    train_labels.append(training_set[index][1])

  train_images = np.array(train_images)
  train_labels = dense_to_one_hot(np.array(train_labels), 19)

  testing_set = addition_with_only_one_x_value(1, "test")

  test_images = []
  test_labels = []

  for index in range(len(testing_set)):
    test_images.append(testing_set[index][0][0].numpy())
    test_labels.append(testing_set[index][1])

  test_images = np.array(test_images)
  test_labels = dense_to_one_hot(np.array(test_labels), 19)

  class DataSets(object):
    pass

  mnist = DataSets()
  mnist.train = DataSet(train_images, train_labels)
  mnist.test = DataSet(test_images, test_labels)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 1568])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 19])

  # Build the graph for the deep net
  y_mlp, keep_prob = deepnn(x)

  batch_number = tf.shape(y_mlp)[0]
  label_examples = tf.greater(tf.reduce_max(y_, axis=1), tf.zeros([batch_number,]))
  label_examples = tf.cast(label_examples, tf.float32)

  with tf.name_scope('cross_entropy'):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_mlp)
    cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_mlp), axis=1)
    
  with tf.name_scope('wmc'):
    normalized_logits = tf.nn.sigmoid(y_mlp)
    wmc_tmp = tf.zeros([batch_number,])
    for i in range(19):
        one_situation = tf.concat(
          [tf.concat([tf.ones([batch_number, i]), tf.zeros([batch_number, 1])], axis=1),
           tf.ones([batch_number, 19-i-1])], axis=1)
        wmc_tmp += tf.reduce_prod(one_situation - normalized_logits, axis=1)
  wmc_tmp = tf.abs(wmc_tmp)
  wmc = tf.reduce_mean(wmc_tmp)

  with tf.name_scope('loss'):
    unlabel_examples = tf.ones([batch_number,]) - label_examples
    log_wmc = tf.log(wmc_tmp)
    loss = -0.0005*tf.multiply(unlabel_examples, log_wmc) - 0.0005*tf.multiply(label_examples, log_wmc) + tf.multiply(label_examples, cross_entropy)
    loss = tf.reduce_mean(loss)
  
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_mlp, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    correct_prediction = tf.multiply(correct_prediction, label_examples)
  accuracy = tf.reduce_sum(correct_prediction) / tf.reduce_sum(label_examples)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_average_accuracy, train_average_wmc, train_average_loss = 0.0, 0.0, 0.0

    epochs = 10
    for t in range(epochs):
      print("Epoch {}\n-------------------------------".format(t+1))

      for i in range(30001):
        images, labels = mnist.train.next_batch(FLAGS.batch_size)
        _, train_accuracy, train_wmc, train_loss =  sess.run([train_step, accuracy, wmc, loss], feed_dict={x: images, y_: labels, keep_prob: 1})
        train_average_accuracy += train_accuracy
        train_average_wmc += train_wmc
        train_average_loss += train_loss

        if i % 100 == 0:
          print(i)
        #   train_average_accuracy /= 100
        #   train_average_wmc /= 100
        #   train_average_loss /= 100
        #   with open("log.txt", 'a') as outFile:
        #     print('step %d, training_accuracy %g, train_loss %g, wmc %g' % (i, train_average_accuracy, train_average_loss, train_average_wmc))
        #     outFile.write('step %d, training_accuracy %g, train_loss %g, wmc %g\n' % (i, train_average_accuracy, train_average_loss, train_average_wmc))
        #     train_average_accuracy, train_average_wmc, train_average_loss = 0.0, 0.0, 0.0
        # if i % 500 == 0:

      test_accuracy = accuracy.eval(feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels,
          keep_prob: 1})
      with open("log.txt", 'a') as outFile:
        print('test accuracy %g' % (test_accuracy))
        outFile.write('test accuracy %g\n' % (test_accuracy))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str,
                      default='mnist_data/',
                      help='Directory for storing input data')
  parser.add_argument('--batch_size', type=int,
                      default='1',
                      help='Batch size for mini-batch Adams gradient descent.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)