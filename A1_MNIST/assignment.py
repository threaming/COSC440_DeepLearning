from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import gzip, os


class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = [28,28] # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.5

        # TODO: Initialize weights and biases
        self.W = np.zeros([self.num_classes,self.input_size[0]*self.input_size[1]])
        self.b = np.zeros([1,self.num_classes])

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: output, unscaled output values for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        return inputs.dot(self.W.T)+self.b

    def back_propagation(self, inputs, outputs, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass. The learning algorithm for updating weights
        and biases is the Perceptron Learning Algorithm discussed in
        lecture (and described in the assignment writeup). This function should
        handle a batch_size number of inputs by taking the average of the gradients
        across all inputs in the batch.
        :param inputs: batch inputs (a batch of images)
        :param outputs: matrix that contains the unscaled output values of each
        class for each image
        :param labels: true labels
        :return: gradient for weights, and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        labels_OH = np.zeros((labels.size, self.num_classes), dtype=int)
        labels_OH[np.arange(labels.size),labels] = 1  # One-Hot encoding
        # y = expected - (f(x)>0)
        predictions = np.argmax(outputs, axis=1)              # 100
        predictions_OH = np.zeros_like(outputs)               # 100x10
        predictions_OH[np.arange(outputs.shape[0]), predictions] = 1
        y = labels_OH - predictions_OH                        # 100x10
        # db = y*1
        gradB = np.mean(y, axis=0)                            # 10
        # dW = y*x
        y = y.reshape((outputs.shape[0],1,self.num_classes))  # 100x1x10
        inputs = inputs.reshape((outputs.shape[0],self.input_size[0]*self.input_size[1],1))     # 100x784x1
        dW = inputs*y                                         # 100x784x10
        gradW = np.mean(dW, axis=0)                           # 784x10
        return gradW.T, gradB          # transpose for assignment_tests to pass

    def accuracy(self, outputs, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param outputs: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        predicted_labels = np.argmax(outputs, axis=1)
        return np.mean(predicted_labels == labels)

    def gradient_descent(self, gradW, gradB):
        """
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        """
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = np.transpose(self.W.T + self.learning_rate*gradW.T)
        self.b = self.b + self.learning_rate*gradB


def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    """
    print(f"Training with batch size {model.batch_size} and {len(train_inputs)} data inputs started...")
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    for start in range(0, len(train_inputs), model.batch_size):
        inputs = train_inputs[start:start+model.batch_size]
        labels = train_labels[start:start+model.batch_size]

        # TODO: For every batch, compute then descend the gradients for the model's weights
        outputs = model.call(inputs)
        gradientsW, gradientsB = model.back_propagation(inputs, outputs, labels)
        model.gradient_descent(gradientsW, gradientsB)
    
    print("Training finished...")


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """

    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    outputs = model.call(test_inputs)
    accuracy = model.accuracy(outputs, test_labels)
    return accuracy

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main(mnist_data_folder):
    """
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    """
    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels = get_data(mnist_data_folder + 'train-images-idx3-ubyte.gz',\
                                          mnist_data_folder + 'train-labels-idx1-ubyte.gz',\
                                          60000)
    test_inputs, test_labels   = get_data(mnist_data_folder + 't10k-images-idx3-ubyte.gz',\
                                          mnist_data_folder + 't10k-labels-idx1-ubyte.gz',\
                                          10000)

    # TODO: Create Model
    badassModel = Model();

    # TODO: Train model by calling train() ONCE on all data
    train(badassModel, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    accuracy = test(badassModel, test_inputs, test_labels)
    print(f"The accuracy is {accuracy*100}%")

    # TODO: Visualize the data by using visualize_results()
    visualize_results(test_inputs[0:10], \
                      badassModel.call(test_inputs[0:10]), \
                      test_labels[0:10])

    print("end of assignment 1")

if __name__ == '__main__':
    #TODO: you might need to change this to something else if you run locally
    main("./MNIST_data")
