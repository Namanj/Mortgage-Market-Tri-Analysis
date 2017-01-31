import pandas as pd
import numpy as np
import random
import cPickle as pickle
import datetime as dt
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

#global global_X, global_y, oversampled_global_X, oversampled_global_y, global_i, global_indices
global_X = None
global_y = None
oversampled_global_X = None
oversampled_global_y = None
global_i = 0
global_indices = []

# class credit_risk_model(object):

#     def __init__(self):
#         '''
#         I've decided to try out Linear Regression and Random Forest
#         '''
#         self._X = None
#         self._y = None

#     def dump_models(self, model_name):
#         if model_name == 'lr':
#             with open('model.pkl', 'w') as f:
#                 pickle.dump(self._lr, f)

#         if model_name == 'rf':
#             with open('model.pkl', 'w') as f:
#                 pickle.dump(self._rf, f)

#     def get_feature_engineered_data(self):
#         '''
#         Uses the distance metric specified
#         INPUT:
#             - X_new: 1D Numpy array, corresponding to 1 entry from X
#             - y_new: 1D Numpy array
#         OUTPUT:
#         Percentage of correct label predictions
#         '''
#         df = pd.read_excel('data/Loan Book Nov-16.xlsx')
#         cols = ['PF','Net PF','Bouncing']

#         df = df[cols]
#         df['Risky'] = df['Bouncing'].map(lambda item: 1 if item >0.01 else 0)
        
#         X = df
#         y = df['Risky']
#         return X, y, cols

#     # def standard_confusion_matrix(self, y_true, y_predict):
#     #     [[tn, fp], [fn, tp]] = metrics.confusion_matrix(y_true, y_predict)
#     #     return np.array([[tp, fp], [fn, tn]])


#     # def profit_curve(self, cost_benefit_matrix, probabilities, y_true):
#     #     thresholds = sorted(probabilities, reverse=True)
#     #     profits = []
#     #     for threshold in thresholds:
#     #         y_predict = probabilities > threshold
#     #         confusion_mat = standard_confusion_matrix(y_true, y_predict)
#     #         profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
#     #         profits.append(profit)
#     #     return thresholds, profits


#     # def run_profit_curve(self, model, costbenefit, X_train, X_test, y_train, y_test):
#     #     model.fit(X_train, y_train)
#     #     probabilities = model.predict_proba(X_test)[:, 1]
#     #     thresholds, profits = profit_curve(costbenefit, probabilities, y_test)
#     #     return thresholds, profits


#     # def plot_profit_models(self, models, costbenefit, X_train, X_test, y_train, y_test):
#     #     percentages = np.linspace(0, 100, len(y_test))
#     #     for model in models:
#     #         thresholds, profits = run_profit_curve(model,
#     #                                                costbenefit,
#     #                                                X_train, X_test,
#     #                                                y_train, y_test)
#     #         plt.plot(percentages, profits, label=model.__class__.__name__)
#     #     plt.title("Profit Curves")
#     #     plt.xlabel("Percentage of test instances (decreasing by score)")
#     #     plt.ylabel("Profit")
#     #     plt.legend(loc='lower right')
#     #     plt.savefig('profit_curve.png')
#     #     plt.show()


#     # def roc_curve(self, probabilities, labels):
#     #     '''
#     #     INPUT: numpy array, numpy array
#     #     OUTPUT: list, list, list
#     #     Take a numpy array of the predicted probabilities and a numpy array of the
#     #     true labels.
#     #     Return the True Positive Rates, False Positive Rates and Thresholds for the
#     #     ROC curve.
#     #     '''

#     #     thresholds = np.sort(probabilities)

#     #     tprs = []
#     #     fprs = []

#     #     num_positive_cases = sum(labels)
#     #     num_negative_cases = len(labels) - num_positive_cases

#     #     for threshold in thresholds:
#     #         # With this threshold, give the prediction of each instance
#     #         predicted_positive = probabilities >= threshold
#     #         # Calculate the number of correctly predicted positive cases
#     #         true_positives = np.sum(predicted_positive * labels)
#     #         # Calculate the number of incorrectly predicted positive cases
#     #         false_positives = np.sum(predicted_positive) - true_positives
#     #         # Calculate the True Positive Rate
#     #         tpr = true_positives / float(num_positive_cases)
#     #         # Calculate the False Positive Rate
#     #         fpr = false_positives / float(num_negative_cases)

#     #         fprs.append(fpr)
#     #         tprs.append(tpr)

#     #     return tprs, fprs, thresholds.tolist()

#     # def lr_metrics(self, r, p, a, f):
#     #     '''
#     #     Input: df, X_train, X_test, y_train, y_test
#     #     Output: model
#     #     Take X_train, X_test, y_train, y_test and fit logistic regression model, perform summary of performance, plot ROC curve, return a model
#     #     '''
    
#     #     print '-' * 20
#     #     #print ':::Logistic Regression Summary:::'
#     #     print 'Recall score: {}'.format(r)
#     #     print 'Precision score: {}'.format(p)
#     #     print 'Accuracy score: {}'.format(a)
#     #     print 'F1 score score: {}'.format(f)
#     #     print ':::Beta Coefficients:::'
#     #     # for name, coef in izip(df.columns, model.coef_[0]):
#     #     #     print "%s: %.4f" % (name, coef)
#     #     print '-' * 20

#     #     # plt.plot(fpr, tpr)
#     #     # plt.xlabel("False Positive Rate (1 - Specificity)")
#     #     # plt.ylabel("True Positive Rate (Sensitivity, Recall)")
#     #     # plt.title("ROC plot of Logistic Regression")
#     #     # plt.show()
#     #     return None

#     # def get_metrics(self, model, X_test, y_test, y_pred):
#     #     # return Recall, Precision, and Accuracy
#     #     return metrics.recall_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), \
#     #             model.score(X_test, y_test), metrics.f1_score(y_test, y_pred)

#     # def rf_metrics(self, X_test, y_test, y_pred):
#     #     print '-' * 20
#     #     print ':::Random Forest Summary:::'
#     #     print 'Recall score: {}'.format(metrics.recall_score(y_test, y_pred))
#     #     print 'Precision score: {}'.format(metrics.precision_score(y_test, y_pred))
#     #     print "Accuracy score:", rf.score(X_test, y_test)
#     #     print "Out of bag score:", rf.oob_score_
#     #     feature_importances = np.argsort(rf.feature_importances_)
#     #     print "Feature importances:", list(df.columns[feature_importances[::-1]])
#     #     print '-' * 20
#     #     return None

#     def oversample(self, X, y = None, target, k = None):
#         """
#         INPUT:
#         X, y - your data in Pandas DataFrame and Series format respectively
#         target - the percentage of positive class
#                  observations in the output
#         OUTPUT:
#         X_oversampled, y_oversampled - oversampled data
#         `oversample` randomly replicates positive observations
#         in X, y to achieve the target proportion
#         """
#         # determine how many new positive observations to generate
#         # replicate randomly selected positive observations
#         # combine new observations with original observations
#         n = X.shape[0]

#         X_minority = X[y == 1]
#         y_minority = y[y == 1]
#         X_maj = X[y == 0]
#         y_maj = y[y == 0]

#         temp_X_min = X_minority.reset_index(drop=True).copy()
#         temp_y_min = y_minority.reset_index(drop=True).copy()

#         ratio = temp_X_min.shape[0] * 1.0 / n

#         while ratio < target:
#             add_index = np.random.choice(temp_X_min.shape[0])
#             temp_X_min = temp_X_min.append(temp_X_min.ix[add_index])
#             temp_y_min = temp_y_min.append(pd.Series(1))
#             temp_X_min.reset_index(drop=True, inplace=True)
#             temp_y_min.reset_index(drop=True, inplace=True)

#             n = n + 1
#             ratio = temp_X_min.shape[0] * 1.0 / n

#         X_oversampled = X_maj.append(temp_X_min)
#         y_oversampled = y_maj.append(temp_y_min)

#         X_oversampled.reset_index(drop=True, inplace=True)
#         y_oversampled.reset_index(drop=True, inplace=True)

#         self._X = X_oversampled
#         self._y = y_oversampled

#         return self._X, self._y

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

# Current Scikit Learn does not support transforming both the dependent and indepdendent variables, thus we cant pass oversample into the pipeline

# class custom_LR(BaseEstimator, TransformerMixin):
#     """Takes in dataframe, extracts road name column, outputs average word length"""
#     def __init__(self, C):
#         self.C = C
#         print self.C
#         self._model = LR()
#         pass

#     def score(self, X, parameters):
#         """The workhorse of this feature extractor"""
#         score = self._model.score(global_X[global_indices[global_i][1]], global_y[global_indices[global_i][1]])
        
#         if global_i == 4:
#             global_i = 0
        
#         return score

#     def fit(self, X, y=None):
#         """Returns `self` unless something different happens in train and test"""
#         self._model.set_params(self.params)
#         self._model.fit(oversampled_global_X, oversampled_global_y)
#         return self   

# class ModelTransformer(TransformerMixin):

#     def __init__(self, model):
#         self.model = model

#     def fit(self, *args, **kwargs):
#         self._model.set_params(self.params)
#         self._model.fit(oversampled_global_X, oversampled_global_y)
#         return self

#     def transform(self, X, **transform_params):
#         score = self.model.score(global_X[global_indices[global_i][1]], global_y[global_indices[global_i][1]])
        
#         if global_i == 4:
#             global_i = 0
        
#         return score

class CustomEstimator(BaseEstimator):

    def __init__(self, C=1, penalty = 'l2'):
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
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # def do_stratified_cv(self, model_name, X_oversampled, y_oversampled, c = 1.0):
    #     '''
    #     Uses the distance metric specified
    #     INPUT:
    #         - X_new: 1D Numpy array, corresponding to 1 entry from X
    #         - y_new: 1D Numpy array
    #     OUTPUT:
    #     Percentage of correct label predictions
    #     '''
    #     skf = StratifiedKFold(y_oversampled, n_folds = 5, random_state = 0)

    #     y_preds, scores = [], []
    #     recall, precision, accuracy, f1_score = [], [], [], []

    #     for train_index, test_index in skf:
    #         X_train_cv, X_test_cv, y_train_cv, y_test_cv = \
    #                     X_oversampled.ix[train_index], X_oversampled.ix[test_index], \
    #                     y_oversampled.ix[train_index], y_oversampled.ix[test_index]

    #         # use these to train and validate model and get a score
    #         model = model_name()
    #         model.fit(X_train_cv, y_train_cv)
    #         y_pred = model.predict(X_test_cv)
            
    #         r, p, a, f = self.get_metrics(model, X_test_cv, y_test_cv, y_pred)
            
    #         recall.append(r)
    #         precision.append(p)
    #         accuracy.append(a)
    #         f1_score.append(f)

    #         y_preds.append(y_pred)
    #         scores.append(model.score(X_test_cv, y_test_cv))

    #     self.lr_metrics(np.mean(recall), np.mean(precision), np.mean(accuracy), np.mean(f1_score))

    #     # return coefficents that work the best
    #     return np.mean(scores), y_preds


    # def scale_features(self, df):
    #     df = df.apply(pd.to_numeric)
    #     return df.apply(lambda x: MinMaxScaler().fit_transform(x))

if __name__ == '__main__':
    #             crm = credit_risk_model()
    # X, y, cols = crm.get_feature_engineered_data()
    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
    #                  columns= iris['feature_names'] + ['target'])    
    # X = data
    # y = data.target
    # y = np.mod(y, 2)

	df = pd.read_excel('data/Loan Book Nov-16.xlsx')
	good_df = df[df['Bouncing'] < 10549.66]

	# good_df = good_df.drop(['App No','Cus No','Loan#','Unnamed: 38', 'Short Term'] , axis = 1)
	good_df['target'] = good_df['Bouncing'] * 1.0
	good_df['target'] = good_df['target'].map(lambda item: item if item >0.001 else 0)
	good_df['Type'] = good_df['Type'].apply(lambda item: 'Downsize' if item == 'downsize' else item)
	good_df['Product Par Name'] = good_df['Product Par Name'].apply(lambda item: 'Home Loan' if 'Home Loan' in item and 'Home Loan Rehab' not in item else item)
	good_df['Name of Proj'] = good_df['Name of Proj'].apply(lambda item: 'Default Builder' if item not in 'DEFAULT PROJECT' else 'DEFAULT PROJECT')
	good_df['Rl/ Urb'] = good_df['Rl/ Urb'].apply(lambda item: 'other' if item not in ' Rural' and item not in 'Urban' else item)
	good_df['count'] = 1

	final_cols = ['Rl/ Urb', 'Name of Proj', 'Product Par Name' , 'Type', 'Final DPD Nov-16', 'target']
	final_df = good_df[final_cols]
	final_df = pd.get_dummies(final_df)

	X = final_df
	y = final_df['target']

    # Split data into training Set and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

            # pipeline = Pipeline([('scaler', crm.scale_features()),
            #                 ('svc', SVC(kernel='linear'))])

            # score, y_pred = crm.do_stratified_cv( LR, X_train, y_train) 
            # X_train_oversampled, y_train_oversampled = crm.oversample( pd.DataFrame(X_train_scaled, columns = cols), pd.Series(y_train), target = 0.3)
            # X_train_scaled = crm.scale_features(X_train_oversampled)

    # Using these global variables is my hack to add oversampling as a step within a pipeline, a functionality not yet implemented in Scikit-learn
    # The pipeline is used to iterate over the hyper-parameter space to find the best coefficients
    global_X = X_train
    global_y = y_train
    global_X.reset_index(drop = True, inplace=True)
    global_y.reset_index(drop = True, inplace = True)
                        # Pipeline = oversample, scale_features, model

                        # skf = StratifiedKFold(y_train, n_folds = 5, random_state = 0)

                        # scores = []
                        #         recall, precision, accuracy, f1_score = [], [], [], []

                        # for train_index, test_index in skf:
                        #     # Have to resample outside the pipeline as the current Scikit-learn version tranformer does not support resampling of the dependent variable
                        #     # X_train_oversampled, y_train_oversampled = oversample(X_train[train_index], y_train[train_index], target = 0.3)
                        #     pipeline.fit(X_train, y_train)
                        #     scores.append(pipeline.score(train[cv_idx], train_y[cv_idx]))

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
    print 'karde bhai'
    print final.score(X_test, y_test)
    # define cost-benefit matrix
    # costbenefit = np.array([[0, -1], [-5, 0]])  # TP=0, FP=-1, FN=-5, TN=0

    # logis_model = lr_model(dfx, X_oversampled, X_test, y_oversampled, y_test)
    # rf_model(dfx, X_oversampled, X_test, y_oversampled, y_test)

    # print 'Generating Profit Curve...'
    # models = [LR(), RF(n_estimators=30, oob_score=True)]
    # plot_profit_models(models, costbenefit, X_oversampled, X_test, y_oversampled, y_test)

    # model = logis_model  # model for pickle
    # with open('model.pkl', 'w') as f:
    #     pickle.dump(model, f)
