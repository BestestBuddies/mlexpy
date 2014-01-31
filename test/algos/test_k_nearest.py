import unittest
from sklearn import datasets
from mlexpy.algos.k_nearest import KNearest


class TestNaiveBayes(unittest.TestCase):
    """ Testing K Nearest Neighbors """

    def setUp(self):
        self.iris = datasets.load_iris()
        self.X = self.iris.data
        self.Y = self.iris.target

    def test_single_row_input_answer(self):
        knn = KNearest(5)
        train = knn.X[0:125]
        test = knn.X[125:]

        knn.fit(train)
        self.assertEqual(knn.predict(test[0]), 2)
        self.assertEqual(knn.predict(test[1]), 1)
        self.assertEqual(knn.predict(test[2]), 1)
        self.assertEqual(knn.predict(test[3]), 2)
        self.assertEqual(knn.predict(test[4]), 2)
