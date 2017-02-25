import pandas as pd
import numpy as np
import random
import datetime as dt
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as AdaBoost

import random
random.seed(0)

global_X = None
global_y = None
oversampled_global_X = None
oversampled_global_y = None

global_i = 0
global_indices = []


# Current Scikit Learn version does not support transforming both the dependent and indepdendent variables, 
# thus we can't pass oversample into the pipeline without a little hack
class oversample(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""
    def __init__(self):
        pass

    def transform(self, X, y = None, target = 0.4):
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
        #oversampled_global_X = oversampled_global_X.apply(pd.to_numeric)
        oversampled_global_X = MinMaxScaler().fit_transform(oversampled_global_X)
        return oversampled_global_X

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class CustomEstimator(BaseEstimator):

    def __init__(self, C = 1, penalty = 'l2'):
        self.C = C
        self.penalty = penalty
        self._model = AdaBoost(n_estimators = 50)
        self._model.set_params(C = C, penalty = penalty)
        pass

    def transform(self, X, y=None):
        return self.score(X)
    
    def predict(self, X):
        return self._model.predict(X)
    
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
    df = pd.read_excel('Loan Book Nov-16.xlsx')
    df = df[:-3]
    df.drop(10787, inplace=True)
    df.reset_index()
    good_df = df[df['Bouncing'] < 10549.66]
    good_df = df

    # good_df = good_df.drop(['App No','Cus No','Loan#','Unnamed: 38', 'Short Term'] , axis = 1)
    good_df.ix[:,'target'] = 1.0 * good_df['Bouncing'].copy()
    good_df.ix[:,'target'] = good_df['target'].map(lambda item: item if item >0.001 else 0)
    good_df.ix[:,'Type'] = good_df['Type'].apply(lambda item: 'Downsize' if item == 'downsize' else item)
    good_df.ix[:,'Product Par Name'] = good_df['Product Par Name'].apply(lambda item: 'Home Loan' if 'Home Loan' in item and 'Home Loan Rehab' not in item else item)
    good_df.ix[:,'Name of Proj'] = good_df['Name of Proj'].apply(lambda item: 'Default Builder' if item not in 'DEFAULT PROJECT' else 'DEFAULT PROJECT')
    good_df.ix[:,'Rl/ Urb'] = good_df['Rl/ Urb'].apply(lambda item: 'other' if item not in ' Rural' and item not in 'Urban' else item)
    #good_df['count'] = 1
    
    good_df['EMI'].fillna(method = 'bfill', inplace = True)
    extra_cols = ['Tenor', 'ROI', 'EMI', 'Rec',  ' Amt Db']
    final_cols = ['Rl/ Urb', 'Name of Proj', 'Product Par Name' , 'Type', 'Final DPD Nov-16', 'target']
    final_df = good_df[final_cols + extra_cols]
    final_df = pd.get_dummies(final_df)
    final_df.ix[:,'target'] = final_df['target'].map(lambda item: 1 if item >= 0.01 else 0)

    X = final_df.copy()
    X.drop('target', axis = 1, inplace = True)

    y = final_df['target']

    # Split data into training Set and Testing Set
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size = 0.3, random_state = 0, stratify = y)

    # Using these global variables is my hack to add oversampling as a step within a pipeline, a functionality not yet implemented in Scikit-learn
    # The pipeline is used to iterate over the hyper-parameter space to find the best coefficients
    global_X = X
    global_y = y
    global_X.reset_index(drop = True, inplace = True)
    global_y.reset_index(drop = True, inplace = True)
    oversampled_global_X = X.drop('target', axis = 1)
    oversampled_global_X = pd.DataFrame(oversampled_global_X)
    oversampled_global_y = y

    skf = StratifiedKFold(n_splits = 5, random_state = 0)
    for train_index, test_index in skf.split(X_train, y_train):
        global_indices.append([train_index, test_index])

    pipeline = Pipeline([
        ('oversample',oversample()),
        ('scale',custom_MinMaxScaler()),
        ('model', CustomEstimator())  # classifier
    ])

    parameters = {'model__n_estimators': [50, 100, 150]}

    final = GridSearchCV(pipeline, parameters, cv = 5, verbose = 1 , scoring='f1')
    final.fit(X_train, y_train)

    print final.best_params_
    print final.best_score_
    y_preds = final.predict(X_test)
    print metrics.classification_report(y_test, y_preds)
    

    