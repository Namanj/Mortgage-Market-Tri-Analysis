import pandas as pd
import numpy as np
import random
import cPickle as pickle

from sklearn import metrics
import matplotlib.pyplot as plt


random.seed(0)

class credit_risk_model(object):

    def __init__(self):
        self._model = None
        pass

    def load_models(self):
        with open('model.pkl', 'r') as f:
            self._model = pickle.load(f)

    def standard_confusion_matrix(self, y_true, y_predict):
        [[tn, fp], [fn, tp]] = metrics.confusion_matrix(y_true, y_predict)
        return np.array([[tp, fp], [fn, tn]])


    def profit_curve(self, cost_benefit_matrix, probabilities, y_true):
        thresholds = sorted(set(probabilities), reverse=True)
        profits = []
        for threshold in thresholds:
            y_predict = probabilities > threshold
            confusion_mat = self.standard_confusion_matrix(y_true, y_predict)
            profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
            profits.append(profit)
        return thresholds, profits


    def run_profit_curve(self, costbenefit, X_train, X_test, y_train, y_test):
        probabilities = self._model.predict_proba(X_test)[:, 1]
        thresholds, profits = self.profit_curve(costbenefit, probabilities, y_test)
        return thresholds, profits, len(set(probabilities))


    def plot_profit_models(self, costbenefit, X_train, X_test, y_train, y_test):
        self.load_models()

        thresholds, profits, length = self.run_profit_curve(
                                               costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        percentages = np.linspace(0, 100, length)

        plt.plot(percentages, profits, label=self._model.__class__.__name__)
        plt.title("Profit Curves")
        plt.xlabel("Percentage of test instances (decreasing by score)")
        plt.ylabel("Profit")
        plt.legend(loc = 'lower right')
        plt.savefig('profit_curve.png')
        plt.show()

