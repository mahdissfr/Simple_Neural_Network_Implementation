import random

# Third-party libraries
import numpy
import numpy as np

from file_handler import read_from_file
from plot import plot, categorize_result


class Network1(object):

    def __init__(self,train, test):
        self.training_data = train
        self.test_data = test
        self.biases = numpy.random.normal(loc=0.0, scale=1.0, size=None)
        self.weights = [numpy.random.normal(loc=0.0, scale=1.0, size=None),
                        numpy.random.normal(loc=0.0, scale=1.0, size=None)]
        self.gradb = 0
        self.gradw = [0, 0]

    def feedforward(self, data):
        input = self.weights[0] * data[0] + self.weights[1] * data[1] + self.biases
        y = sigmoid(input)
        return y

    def reset_grad(self):
        self.gradb = 0
        self.gradw = [0, 0]

    def SGD(self, epochs, lr):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n_test = len(self.test_data)
        # n = len(self.training_data)
        for j in range(epochs):
            self.reset_grad()
            for row in self.training_data:
                self.backprop(row)
            self.update_mini_batch(lr)
            # print("Epoch {0}: {1}".format(j, self.evaluate(self.test_data) * 100 / n_test))
        print("Epoch {0}: {1}".format(j, self.evaluate(self.test_data) * 100 / n_test))
        result = []
        for data in self.test_data:
            x = 0
            if self.feedforward(data) > 0.5:
                x = 1
            result.append([data[0], data[1], x])

        x0, y0, x1, y1 = categorize_result(result)
        plot(x0, y0, x1, y1)




    def update_mini_batch(self, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        lr = eta / len(self.training_data)
        self.weights[0] -= lr * self.gradw[0]
        self.weights[1] -= lr * self.gradw[1]
        self.biases -= lr * self.gradb

    def backprop(self, data):
        y = self.feedforward(data)
        self.gradb += (y - data[2]) * y * (1 - y)
        self.gradw[0] += data[0] * self.gradb
        self.gradw[1] += data[1] * self.gradb

    def evaluate(self, test_data):
        test_results = []
        for data in test_data:
            x = 0
            if self.feedforward(data) > 0.5:
                x = 1
            test_results.append([x, data[2]])

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, y, y0):
        return 0.5 * pow(y - y0, 2)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))



