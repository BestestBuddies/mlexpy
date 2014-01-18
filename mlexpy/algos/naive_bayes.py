"""
This module implements naive bayes
"""
import pandas as pd

class NaiveBayes(object):

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def fit(self):
        self.probs = {}
        result = self.data['Play']
        total_yes = result[result == 'Yes'].count()
        total = result.count()
        self.prob_yes = (total_yes * 1.0) / (total)

        for header in self.data:  # loops through sunny, temp, play, etc
            if header == 'Play':
                continue
            self.probs[header] = {}
            for value in self.data[header].unique():
                count_yes = self.data[(self.data[header] == value) & (self.data['Play'] == 'Yes')].count()[0]
                count_total = self.data[(self.data[header] == value)].count()[0]
                self.probs[header][value] = (1.0 *count_yes) / count_total


nb = NaiveBayes('tennis.csv')
nb.fit()

