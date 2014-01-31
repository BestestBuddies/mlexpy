"""
This module implements the k-nearest neighbor algorithm
"""

from math import sqrt
from sklearn import datasets
from collections import Counter

class KNearest():

    def __init__(self, k):
        self.iris = datasets.load_iris()
        self.X = self.iris.data
        self.Y = self.iris.target
        self.train = None
        self.k = k

    def fit(self, train):
        self.train = train

    def predict(self, sample):
        """
        assumes single sample is input
        """
        distance = {}
        for row_num, row in enumerate(self.train):  # Loops through rows to calculate each distance
            dist = 0
            for value_num, value in enumerate(row):
                dist += (value - sample[value_num]) ** 2
            dist = sqrt(dist)
            distance[row_num] = dist
        distance_sorted = sorted(distance.items(), key=lambda x: x[1])[0:self.k] # List of tuples where [0] is rownum, [1] is dist
        outputs = []
        for pair in distance_sorted:
            outputs.append(self.Y[pair[0]])  # result of that entry
        counts = Counter(outputs).most_common()  # list of most common outputs in order
        return counts[0][0] # So messy clean dis shit up

knn = KNearest(5)
train = knn.X[0:125]
test = knn.X[125:]

knn.fit(train)
print knn.predict(test[0])
print knn.predict(test[1])
print knn.predict(test[2])
print knn.predict(test[3])
print knn.predict(test[4])
print knn.predict(test[5])
