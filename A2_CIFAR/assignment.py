from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random


def linear_unit(x, W, b):
  return tf.matmul(x, W) + b

class ModelPart0:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.trainable_variables = [self.W1, self.B1]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		    # this reshape "flattens" the image data
        inputs = np.reshape(inputs, [inputs.shape[0],-1])
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        x = linear_unit(inputs, self.W1, self.B1)
        return x

class ModelPart1:
    def __init__(self):
        """
        This model class contains a dual layer network for slightly better performance.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        inner_connect = 256
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, inner_connect],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, inner_connect],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")
        self.W2 = tf.Variable(tf.random.truncated_normal([inner_connect, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.trainable_variables = [self.W1, self.B1, self.W2, self.B2]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		    # this reshape "flattens" the image data
        inputs = np.reshape(inputs, [inputs.shape[0],-1])
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        x1 = tf.nn.relu(linear_unit(inputs, self.W1, self.B1))
        x2 = linear_unit(x1, self.W2, self.B2)
        return x2

class ModelPart3:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        connect_input = 1728
        output = 2

        self.fB1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, 9],
								                                        dtype=tf.float32,
								                                        stddev=0.1),
								              name="fB1")
        self.fB2 = tf.Variable(tf.random.truncated_normal([3, 3, 9, 27],
								                                        dtype=tf.float32,
								                                        stddev=0.1),
								              name="fB2")
        self.W1 = tf.Variable(tf.random.truncated_normal([connect_input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.trainable_variables = [self.fB1, self.fB2, self.W1, self.B1]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		    # this reshape "flattens" the image data
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        x1 = tf.nn.relu(tf.nn.conv2d(inputs,self.fB1,strides=[1,1,1,1],padding='SAME')) 
        x2 = tf.nn.max_pool2d(x1,ksize=(2,2),strides=[1,2,2,1],padding='VALID')         
        x3 = tf.nn.relu(tf.nn.conv2d(x2,self.fB2,strides=[1,1,1,1],padding='SAME'))
        x4 = tf.nn.max_pool2d(x3,ksize=(2,2),strides=[1,2,2,1],padding='VALID')
        x4 = tf.reshape(x4, [x4.shape[0],-1])
        x5 = linear_unit(x4, self.W1, self.B1)
        return x5

def loss(logits, labels):
  """
	Calculates the cross-entropy loss after one forward pass.
	:param logits: during training, a matrix of shape (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	Softmax is applied in this function.
	:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
	:return: the loss of the model as a Tensor
	"""
  loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
	"""
	Calculates the model's prediction accuracy by comparing
	logits to correct labels – no need to modify this.
	:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

	NOTE: DO NOT EDIT

	:return: the accuracy of the model as a Tensor
	"""
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
  '''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
	and labels - ensure that they are shuffled in the same order using tf.gather.
	You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
  ''' 
  # Shuffle Data
  idx = np.arange(train_inputs.shape[0])
  shuffle_idx = tf.random.shuffle(idx)
  shuffled_inputs = tf.gather(train_inputs, shuffle_idx)
  shuffled_labels = tf.gather(train_labels, shuffle_idx)

  # Batch training
  for start in range(0, len(train_inputs), model.batch_size):
    inputs = shuffled_inputs[start:start+model.batch_size]
    labels = shuffled_labels[start:start+model.batch_size]

    # Implement backprop:
    with tf.GradientTape() as tape:
      predictions = model.call(inputs) # this calls the call function conveniently
      loss_taped = loss(predictions, labels)

    gradients = tape.gradient(loss_taped, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  pass

def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


CLASS_CAT = 3
CLASS_DOG = 5
def main(cifar10_data_folder):
  '''
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
	test your model for a number of epochs. We recommend that you train for
	25 epochs.
	You should receive a final accuracy on the testing examples for cat and dog
	of ~60% for Part1 and ~70% for Part3.
	:return: None
	'''
  # get data
  train_inputs, train_labels = get_data(cifar10_data_folder + 'train', CLASS_DOG, CLASS_CAT)
  test_inputs, test_labels = get_data(cifar10_data_folder + 'test', CLASS_DOG, CLASS_CAT)

  # Train Model
  model = ModelPart3()
  print(f"Training {type(model).__name__}")
  for i in range(25):
    train(model, train_inputs, train_labels)
    # Evaluate accuracy after each epoch
    accuracy = test(model, train_inputs, train_labels)
    print(f"Accuracy on training set after {i+1} Epochs: {accuracy}")
  accuracy = test(model, test_inputs, test_labels)
  print("Training finished...")
  print(f"Accuracy on test set for {type(model).__name__} is {accuracy}")
  visualize_results(test_inputs[:10],model.call(test_inputs[:10]),test_labels[:10],'D','C')
  return


if __name__ == '__main__':
    cifar_data_folder = './CIFAR_data/'
    main(cifar_data_folder)
