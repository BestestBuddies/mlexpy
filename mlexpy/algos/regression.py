"""
This module implements linear regression.
"""
import numpy as np
from numpy import float32
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class LinearRegression(object):
    @classmethod
    def load_data(cls, filename):
        """
        Reads in two files, first being your input features, second being the known output value. Data expected and method can be found
        at http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html

        :returns weights of theta vector corresponding to fit function
        """
        X = []
        y = []
        for line in open(filename):
            rows = [int(x) for x in line.split(',')]
            X.append(rows[0:2])
            y.append(rows[-1])
        X = np.array(X, float32)
        y = np.array(y, int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #min_max_scalar = MinMaxScaler()
        #X_scaled = min_max_scalar.fit_transform(X)

        X = np.concatenate((np.ones((X.shape[0], 1)), X_scaled), axis=1)
        cls.fit(np.array(X, float32), np.array(y, int))

    @classmethod
    def fit(cls, X, Y):
        """ Regresses data in a numpy array format matrix([1,2,3], [4,5,6], [7,8,9], [9,10,11])

        :returns an n dim list of the weights of the associated weight vector
        """
        cls.num_samples, cls.num_features = X.shape
        theta = np.zeros(cls.num_features)

        cls.learn_rate = 1
        for i in range(0, 100):
            print cls.calculate_cost(theta, X, Y)
            print theta
            theta = cls.update_values(theta, X, Y)

    @classmethod
    def calculate_cost(cls, theta, X, Y):
        """
        Calculates cost function of linear regression
        """

        error = cls.calc_hyp(theta, X) - Y
        sq_error = np.dot(error.transpose(), error) / (2 * cls.num_samples)
        return float(sq_error)

    @classmethod
    def update_values(cls, theta, X, Y):
        """
        Updates the theta vector for current iteration based on gradient descent

        :returns a copy of the new theta vector
        """
        error = cls.calc_hyp(theta, X) - Y
        gradient = np.dot(X.transpose(), error) / cls.num_samples
        theta = theta - (cls.learn_rate  * gradient)
        return theta

    @classmethod
    def calc_hyp(cls, theta, X):
        """
        Calculates hypothesis function
        """

        return np.dot(X, theta)

    @classmethod
    def fit_2d(cls, input_):
        """ Regresses a set of 2d tuples.

        >>> LinearRegression.fit_2d([(0, 1), (1, 0)])
        [(0, 1), (1, 0)]
        >>> LinearRegression.fit_2d([(0, 1), (0, 2), (2, 1), (2, 2)])
        [(0, 1.5), (1, 1.5)]

        See: http://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line

        :param list input_: an array containing tuples of (x, y) coordinate
         pairs
        :returns: An array containing two tuples, one at x = 0, the other at
         x = 1.
        :rtype: list(tuple, tuple)
        """
        b = cls._beta(input_)
        a = cls._alpha(input_, b)
        y_0 = a + (b * 0.)
        y_1 = a + (b * 1.)

        return [(0., y_0), (1., y_1)]

    @classmethod
    def _beta(cls, input_):
        """
        Finds the beta (slope) of the input.

        :param list input_: an array containing tuples of (x, y) coordinate
         pairs
        :returns: the slope of the input
        :rtype: float
        """
        numerator = cls._xybar(input_) - (cls._xbar(input_) * cls._ybar(input_))
        xbar = cls._xbar(input_)
        denominator = cls._xsquarebar(input_) - (xbar * xbar)

        if denominator == 0:
            return None

        return numerator / denominator

    @classmethod
    def _alpha(cls, input_, beta_):
        """
        Finds the alpha (intercept) of the input and beta values.

        :param list input_: an array containing tuples of (x, y) coordinate
         pairs
        :param float beta_: the beta (slope) of the input
        :returns: the intercept of the input
        :rtype: float
        """
        return cls._ybar(input_) - (beta_ * cls._xbar(input_))

    @classmethod
    def _xbar(cls, input_):
        s = 0
        for tup in input_:
            s += tup[0]
        return float(s)/len(input_)

    @classmethod
    def _ybar(cls, input_):
        s = 0
        for tup in input_:
            s += tup[1]
        return float(s)/len(input_)

    @classmethod
    def _xybar(cls, input_):
        s = 0
        for tup in input_:
            s += tup[0] * tup[1]
        return float(s)/len(input_)

    @classmethod
    def _xsquarebar(cls, input_):
        s = 0
        for tup in input_:
            s += tup[0] * tup[0]
        return float(s)/len(input_)

if __name__ == '__main__':
    filename = '../../ex1data2.txt'
    LinearRegression.load_data(filename)
