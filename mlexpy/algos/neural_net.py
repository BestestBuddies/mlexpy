"""
This module implements a neural network algorithm
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class NeuralNet():

    def __init__(self, train_file, valid_file, test_file):  # load data
        self.train = pd.read_csv(train_file, delimiter='\t', header=None)
        self.valid = pd.read_csv(valid_file, delimiter='\t', header=None)
        self.test = pd.read_csv(test_file, delimiter='\t', header=None)
        num_outputs = 3
        self.num_features = self.train.shape[1] - num_outputs
        self.data = {
            'train': {
                'inputs': self.train[range(self.num_features)],
                'outputs': self.train[range(self.num_features, self.num_features + num_outputs)].values,
            },
            'valid': {
                'inputs': self.valid[range(self.num_features)],
                'outputs': self.valid[range(self.num_features, self.num_features + num_outputs)].values,
            },
            'test': {
                'inputs': self.test[range(self.num_features)],
                'outputs': self.test[range(self.num_features, self.num_features + num_outputs)].values,
            },
        }
        self.weights = np.ones((self.num_features+1, num_outputs))

    def initialize_weights(self, max_):
        rnd = random.Random()
        func_ = np.vectorize(lambda x: rnd.uniform(-max_, max_))
        self.weights = func_(self.weights)

    def train_model(self):
        self.initialize_weights(.5)
        errors = {'train': [], 'valid': [], 'test': [], }
        for i in range(500):
            self.back_prop()
            for dataset in errors.keys():
                errors[dataset].append(
                    self.calculate_network_error(self.data[dataset]['outputs'], self.data[dataset]['inputs'])[0]
                )
        self.plot_errors(errors)

    def plot_errors(self, errors):
        plt.plot(errors['train'])
        plt.plot(errors['valid'])
        plt.plot(errors['test'])
        plt.show()

    def calculate_network_error(self, known_outputs, inputs):
        net, predictions = self.feed_forward(inputs)
        #prediction_classes = self.matrix_to_output(predictions)
        #known_classes = self.matrix_to_output(known_outputs)

        known_outputs_list = known_outputs.tolist()
        predictions = predictions.tolist()
        error = sum([(target - output)**2 for target, output in zip(known_outputs, predictions)])
        error *= (1.0 / (len(inputs) * self.num_features))

        #class_error = len([1 for i, j in zip(known_classes, prediction_classes) if i != j])
        #class_error *= (1.0 / known_outputs.shape[0])
        class_error = 0
        return error, class_error

    def feed_forward(self, inputs):
        bias = np.matrix([1] * inputs.shape[0]).T
        biased_input = np.append(inputs, bias, axis=1)  # adds bias matrix as end column
        net = biased_input * self.weights
        predictions = self.compute_activation(net)
        return net, predictions

    def back_prop(self):
        return self.weights

    @classmethod
    def matrix_to_output(cls, matrix_):
        """
        Takes matix like
        1 0 0
        0 1 0
        0 0 1

        :returns matrix of form
        1
        2
        3

        representing class outputs
        """
        matrix_ = matrix_.tolist()
        output_matrix = []
        for line in matrix_:  # should be like [0, 0, 1]
            output_matrix.append([line.index(1)])
        return np.matrix(output_matrix)  # column vector of class values

    @classmethod
    def output_to_matrix(cls, output):
        """
        Takes matix like
        1
        2
        3

        :returns matrix of form
        1 0 0
        0 1 0
        0 0 1

        representing outputs
        """
        output = output.tolist()
        matrix_ = []
        for line in output:  # Will be something like [2]
            entry = [0, 0, 0]
            entry[output[0]-1] = 1
            matrix_.append(entry)
        return matrix_

    @classmethod
    def compute_activation(cls, matrix_):
        func_ = np.vectorize(lambda x: (np.tanh(x) + 1.0) / 2)
        return func_(matrix_)

    @classmethod
    def compute_activation_deriv(cls, matrix_):
        func_ = np.vectorize(lambda x: (1.0 - np.tanh(x) ** 2) / 2)
        return func_(matrix_)


Nn = NeuralNet('iris_training.dat', 'iris_validation.dat', 'iris_test.dat')
Nn.train_model()



