"""
This module implements linear regression.
"""
import numpy as np

class LinearRegression(object):

    @classmethod
    def load_data(cls, filename1, filename2):
        """
        Reads in two files, first being your input features, second being the known output value. Data expected and method can be found
        at http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html

        :returns weights of theta vector corresponding to fit function
        """
        f1 = open(filename1)
        f2 = open(filename2)
        X = []
        Y = []
        for line in f1:
            X.append([float(entry) for entry in line.split()])
        for entry in X:
            entry.insert(0, 1)
        X = np.matrix(X)
        for line in f2:
            Y.append([float(line)])
        Y = np.matrix(Y)
        cls.fit(X, Y)

    @classmethod
    def fit(cls, X, Y):
        """ Regresses data in a numpy array format matrix([1,2,3], [4,5,6], [7,8,9], [9,10,11])

        :returns an n dim list of the weights of the associated weight vector
        """
        cls.num_features = X.shape[1]
        cls.num_samples = X.shape[0]
        theta = []
        for i in range(cls.num_features):
            theta.append(0)
        theta = np.array(theta)
        cls.learn_rate = 0.01
        for i in range(0, 50):
            print cls.calculate_cost(theta, X, Y)
            print theta
            theta = cls.update_values(theta, X, Y)

    @classmethod
    def calculate_cost(cls, theta, X, Y):
        """
        Calculates cost function of linear regression
        """

        hyp = cls.calc_hyp(theta, X)
        return (1.0 / (2 * cls.num_samples)) * sum( [ (hyp[i] - Y[i])**2 for i in range(cls.num_samples) ] )

    @classmethod
    def update_values(cls, theta, X, Y):
        """
        Updates the theta vector for current iteration based on gradient descent

        :returns a copy of the new theta vector
        """
        theta_new = theta.copy()
        hyp = cls.calc_hyp(theta, X)
        for j in range(cls.num_features):
            sum_term = 0.0
            for i in range(cls.num_samples):
                sum_term += float((hyp[i] - Y[i]) * X[i, j])
            sum_term *= (cls.learn_rate / cls.num_samples)
            theta_new[j] = float(theta[j] - sum_term)
        return theta_new

    @classmethod
    def calc_hyp(cls, theta, X):
        """
        Calculates hypothesis function
        """

        hyp = []
        for row in X:
            hyp.append([float(theta * row.transpose())])
        return np.array(hyp)

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
    filename1 = '../../ex3x.dat'
    filename2 = '../../ex3y.dat'
    LinearRegression.load_data(filename1, filename2)
