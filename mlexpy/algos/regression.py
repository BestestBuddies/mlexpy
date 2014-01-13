"""
This module implements linear regression.
"""


class LinearRegression(object):

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

