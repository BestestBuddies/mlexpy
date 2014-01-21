import unittest
from mlexpy.algos.naive_bayes import NaiveBayes
import pandas as pd


class TestNaiveBayes(unittest.TestCase):
    """ Testing Naive Bayes """

    def setUp(self):
        self.nb = NaiveBayes('tennis.csv')
        self.nb.fit()
        dict_input = {'Sky': 'Sunny', 'Temp': 'Hot', 'Humid': 'High', 'Wind': 'Weak'}
        self.input_ = pd.Series(dict_input)

    def test_single_row_input_prob(self):
        self.assertAlmostEqual(self.nb.predict(self.input_, probability=True), 0.241975, places=6)  # hand calculated probability

    def test_single_row_input_answer(self):
        self.assertEqual(self.nb.predict(self.input_), "No")  # hand calculated probability
