import unittest
from mlexpy.algos.regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
    """ Testing Linear Regression """

    def setUp(self):
        pass

    def test_2pt(self):
        input_ = [(0, 1), (1, 0)]
        expected = input_
        actual = LinearRegression.fit_2d(input_)
        self.assertEqual(expected, actual)

    def test_4pt(self):
        input_ = [(0, 1), (0, 2), (2, 1), (2, 2)]
        expected = [(0, 1.5), (1, 1.5)]
        actual = LinearRegression.fit_2d(input_)
        self.assertEqual(expected, actual)

    def test_ml_class_file(self):
        filename1 = '../ex3x.dat'
        filename2 = '../ex3y.dat'
        LinearRegression.load_data(filename1, filename2)
