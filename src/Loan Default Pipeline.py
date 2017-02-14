import pandas as pd
import numpy as np
import random
import cPickle as pickle
from datetime import date

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import SGDClassifier as SGD
from sklearn import metrics

from determine_best_threshold import determine_best_threshold

random.seed(0)

def clean_df(df):
    """
    Accepts raw data and prepares it for the model

    Parameters
    ----------
    df : pandas DataFrame
        Original DataFrame
    """
    df = df[:-3]
    df.drop(10787, inplace=True)
    df.reset_index()
    good_df = df[df['Bouncing'] < 10549.66]
    good_df = df

    good_df.ix[:,'target'] = 1.0 * good_df['Bouncing'].copy()
    good_df.ix[:,'Type'] = good_df['Type'].apply(lambda item: 'Downsize' if item == 'downsize' else item)
    good_df.ix[:,'Product Par Name'] = good_df['Product Par Name'].apply(lambda item: 'Home Loan' if 'Home Loan' in item and 'Home Loan Rehab' not in item else item)
    good_df.ix[:,'Name of Proj'] = good_df['Name of Proj'].apply(lambda item: 'Default Builder' if item not in 'DEFAULT PROJECT' else 'DEFAULT PROJECT')
    good_df.ix[:,'Rl/ Urb'] = good_df['Rl/ Urb'].apply(lambda item: 'other' if item not in ' Rural' and item not in 'Urban' else item)
    
    good_df['EMI'].fillna(method = 'bfill', inplace = True)
    extra_cols = ['Tenor', 'ROI', 'EMI', 'Rec',  ' Amt Db']
    final_cols = ['Rl/ Urb', 'Name of Proj', 'Product Par Name' , 'Type', 'Final DPD Nov-16', 'target']
    final_df = good_df[final_cols ]
    final_df = pd.get_dummies(final_df)
    final_df.ix[:,'target'] = final_df['target'].map(lambda item: 1 if item >= 0.01 else 0)

    X = final_df.copy()
    X.drop('target', axis = 1, inplace = True)

    y = final_df['target']

    d1 = date(2016,12, 1)   
    delta = d1 - good_df['San Date']
    delta = delta.map(lambda item: item.days)

    X['days'] = delta

    # These degrees of freedom were hand tuned
    weights = 1000 + 10000000.0 * (good_df['Bouncing']/ (1.0 * good_df['Loan Amt'] * delta))

    return X, y, weights

def dump_models(model):
    """
    Dump a tuned model into a Pickle file

    Parameters
    ----------
    model : sklearn estimator object
        Model tuned on the data and whose hyperparameters have been tuned
    """
    with open('model.pkl', 'w') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    # Get raw data
    df = pd.read_excel('data/Loan Book Nov-16.xlsx')

    X, y, weights = clean_df(df)

    # Split data into training Set and Testing Set
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, 
                                                    test_size = 0.3, random_state = 0, stratify = y)

    pipeline = Pipeline([
        ('rescale',MinMaxScaler()),
        # ('model', SGD(loss = 'modified_huber',class_weight = 'balanced', penalty = None))  # classifier
        # ('model', RF(class_weight = 'balanced_subsample'))  # classifier
        ('model', AdaBoost(n_estimators = 50))  # classifier
    ])
    
    parameters = {'model__n_estimators': [50, 100, 150]}

    best_model = GridSearchCV(pipeline, parameters, cv = 5, verbose = 1 , scoring = 'f1'
                , fit_params = {'model__sample_weight': np.array(weights[train_indices])})
    best_model.fit(X_train, y_train)

    y_pred_probs = best_model.predict_proba(X_test)
    y_pred_probs = pd.DataFrame(y_pred_probs).ix[ : ,1]
    threshold = 0.48062

    y_preds = y_pred_probs > threshold

    print metrics.classification_report(y_test, y_preds) 
    
    dump_models(best_model)

    dbt = determine_best_threshold()
    costbenefit = np.array([[0, -1], [-43, 0]])  # TP=0, FP=-1, FN=-43, TN=0
    dbt.plot_profit_models(costbenefit, X_train, X_test, y_train, y_test)

