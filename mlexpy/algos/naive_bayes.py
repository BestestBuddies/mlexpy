"""
This module implements naive bayes
"""
import pandas as pd

class NaiveBayes(object):

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.probs_cond = {}  # calculates p('Sunny' | play='Yes') etc
        self.probs = {}  # calculates p('Sunny')
        self.prob_yes = 0.0  # p(play = 'Yes')

    def fit(self):
        """
        Calculates probabilities based on input data and stores them in an internal dictionary
        for predict to use
        """
        result = self.data['Play']
        total_yes = result[result == 'Yes'].count()
        total = result.count()
        self.prob_yes = (total_yes * 1.0) / total

        for header in self.data:  # loops through sunny, temp, play, etc
            if header == 'Play':
                continue
            self.probs_cond[header] = {}
            self.probs[header] = {}
            for value in self.data[header].unique():
                count_yes = self.data[(self.data[header] == value) & (self.data['Play'] == 'Yes')].count()[0]  # p(sunny|yes)
                count_total = self.data[(self.data[header] == value)].count()[0] # p(Sunny)
                self.probs_cond[header][value] = (1.0 * count_yes) / total_yes
                self.probs[header][value] = (1.0 * count_total) / total

    def predict(self, input_, probability=False):
        """
        Takes in a pandas series

        :returns Yes or No to play tennis, or probability of yes if probability=True is passed

        :TODO make this more general and multiple lines of input, different types of structures, etc
        """
        prob_yes = self.prob_yes
        for header, value in input_.iteritems():
            prob_yes *= (self.probs_cond[header][value])
            prob_yes /= (self.probs[header][value])
        if probability:
            return prob_yes
        else:
            return "Yes" if prob_yes > 0.5 else "No"

nb = NaiveBayes('tennis.csv')
nb.fit()
dict_input = {'Sky': 'Sunny', 'Temp': 'Hot', 'Humid': 'High', 'Wind': 'Weak'}
input_ = pd.Series(dict_input)
print nb.predict(input_, probability=True)
