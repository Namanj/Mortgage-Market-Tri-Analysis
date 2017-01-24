import pandas as pd
import numpy as np
import random
import cPickle as pickle
import datetime as dt

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV

from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.ensemble import RandomForestClassifier as RF


random.seed(0)

global_X = None
global_y = None
oversampled_global_X = None
oversampled_global_y = None
global_i = 0
global_indices = []


# Current Scikit Learn does not support transforming both the dependent and indepdendent variables, thus we cant pass oversample into the pipeline
class oversample(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""
    def __init__(self):
        pass

    def transform(self, X, y = None, target = 0.8):
        """The workhorse of this feature extractor"""
        global global_X, global_y, oversampled_global_X, oversampled_global_y, global_i, global_indices

        X = pd.DataFrame(global_X.ix[global_indices[global_i][0]])
        y = pd.Series(global_y.ix[global_indices[global_i][0]])

        n = X.shape[0]

        X_minority = X[y == 1]
        y_minority = y[y == 1]
        X_maj = X[y == 0]
        y_maj = y[y == 0]

        temp_X_min = X_minority.reset_index(drop=True).copy()
        temp_y_min = y_minority.reset_index(drop=True).copy()

        ratio = temp_X_min.shape[0] * 1.0 / n

        while ratio < target:
            add_index = np.random.choice(temp_X_min.shape[0])
            temp_X_min = temp_X_min.append(temp_X_min.ix[add_index])
            temp_y_min = temp_y_min.append(pd.Series(1))
            temp_X_min.reset_index(drop=True, inplace=True)
            temp_y_min.reset_index(drop=True, inplace=True)

            n += 1
            ratio = temp_X_min.shape[0] * 1.0 / n

        X_oversampled = X_maj.append(temp_X_min)
        y_oversampled = y_maj.append(temp_y_min)

        X_oversampled.reset_index(drop=True, inplace=True)
        y_oversampled.reset_index(drop=True, inplace=True)
        X_oversampled.drop('target', axis = 1, inplace = True)

        oversampled_global_X = X_oversampled
        oversampled_global_y = y_oversampled

        global_i +=  1
        # print global_i

        return oversampled_global_X

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class custom_MinMaxScaler(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""
    def __init__(self):
        pass

    def transform(self, X):
        """The workhorse of this feature extractor"""
        global oversampled_global_X
        oversampled_global_X = oversampled_global_X.apply(pd.to_numeric)
        oversampled_global_X = MinMaxScaler().fit_transform(oversampled_global_X)
        return oversampled_global_X

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class CustomEstimator(BaseEstimator):

    def __init__(self, C = 1, penalty = 'l2'):
        self.C = C
        self.penalty = penalty
        self._model = LR()
        self._model.set_params(C = C, penalty = penalty)
        pass

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.score(X)

    def score(self, X, y=None):
        global global_X, global_y, oversampled_global_X, oversampled_global_y, global_i, global_indices

        temp_X = global_X.copy()
        temp_X.drop('target', axis = 1, inplace = True)
        score = self._model.score(temp_X.ix[global_indices[global_i][1]], global_y.ix[global_indices[global_i][1]])
        
        if global_i == 4:
            global_i = 0
        
        return score

    def fit(self, X, y=None):
        self._model.fit(oversampled_global_X, oversampled_global_y)
        return self
       
if __name__ == '__main__':
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])    
    X = data
    y = data.target
    y = np.mod(y, 2)

    # Split data into training Set and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

    # Using these global variables is my hack to add oversampling as a step within a pipeline, a functionality not yet implemented in Scikit-learn
    # The pipeline is used to iterate over the hyper-parameter space to find the best coefficients
    global_X = X_train
    global_y = y_train
    global_X.reset_index(drop = True, inplace=True)
    global_y.reset_index(drop = True, inplace = True)

    pipeline = Pipeline([
        ('oversample',oversample()),
        ('scale',custom_MinMaxScaler()),
        ('model', CustomEstimator())  # classifier
    ])

    parameters = {'model__C': [0.1, 1, 10],
                  'model__penalty': ['l2', 'l1']}

    skf = StratifiedKFold(y_train, n_folds = 5, random_state = 0)
    for train_index, test_index in skf:
        global_indices.append([train_index, test_index])

    final = GridSearchCV(pipeline, parameters)
    final.fit(X_train, y_train)

    print final.best_params_
    print final.best_score_
