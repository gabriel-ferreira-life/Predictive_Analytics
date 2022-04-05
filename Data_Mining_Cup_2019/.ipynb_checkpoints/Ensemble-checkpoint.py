import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.metrics import confusion_matrix, classification_report, make_scorer

def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], columns = dictionary.keys())


def ensemble(RF_val_pred, SVC_val_pred, ADA_val_pred, XGBoost_val_pred, Y, RF_test_pred, SVC_test_pred, ADA_test_pred, XGBoost_test_pred):
    
    ## Defining the input variables 
    X = pd.concat([RF_val_pred, SVC_val_pred, ADA_val_pred, XGBoost_val_pred], axis = 1)
    X.columns = ['RF', 'SVC', 'ADA', 'XGB']
#    Y = Y_val
    X_to_score = pd.concat([RF_test_pred, SVC_test_pred, ADA_test_pred, XGBoost_test_pred], axis = 1)
    X_to_score.columns = ['RF', 'SVC', 'ADA', 'XGB']
    
    ## Splitting the data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
    
    ## Number of trees in random forest
    n_estimators = [100, 300, 500]

    ## Number of features to consider at every split
    max_features = [2, 3]

    ## Maximum number of levels in tree
    max_depth = [3, 5]

    ## Minimum number of samples required to split a node
    min_samples_split = [10, 15]

    ## Minimum number of samples required at each leaf node
    min_samples_leaf = [5, 7]

    ## Creating the dictionary of hyper-parameters
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}

    param_grid = expand_grid(param_grid)

    ## Adding evaluation columns
    param_grid['evaluation'] = np.nan

    for i in range(param_grid.shape[0]):
        print('Working on job', i + 1, 'out of ', param_grid.shape[0])
        ## Fitting the model (using the ith combination of hyper-parameters)
        RF_md = RandomForestClassifier(n_estimators = param_grid['n_estimators'][i],
                                       max_features = param_grid['max_features'][i],
                                       max_depth = param_grid['max_depth'][i],
                                       min_samples_split = param_grid['min_samples_split'][i],
                                       min_samples_leaf = param_grid['min_samples_leaf'][i])

        RF_md.fit(X_train, Y_train)

        ## Predicting on the val dataset
        Y_pred = RF_md.predict_proba(X_test)[:, 1]
        
    ## Defining cutoff values in a data-frame
    results = pd.DataFrame({'cutoffs': np.round(np.linspace(0.05, 0.95, num = 40, endpoint = True), 2)})
    results['cost'] = np.nan
        
    ## Changing likelihoods to labels
    for i in range(0, results.shape[0]):
            
        ## Changing likelihoods to labels
        Y_pred_lab = np.where(Y_pred < results['cutoffs'][i], 0, 1)
        
        ## Computing confusion matrix and scoring based on description
        X = confusion_matrix(Y_pred_lab, Y_test)
        results['cost'][i] = -25 * X[1, 0] - 5 * X[0, 1] + 5 * X[1, 1]
        
    ## Sorting results 
    results = results.sort_values(by = 'cutoffs', ascending = False).reset_index(drop = True)
    print("Cutoff: ", results['cutoffs'][0])
    print("Score: ", results['cost'][0])

    
    ## Build the model to score the test
    RF = RandomForestClassifier(n_estimators = param_grid['n_estimators'][0],
                                max_features = param_grid['max_features'][0],
                                max_depth = param_grid['max_depth'][0],
                                min_samples_split = param_grid['min_samples_split'][0],
                                min_samples_leaf = param_grid['min_samples_leaf'][0]).fit(X_train, Y_train)


    ## Predicting on the dataset to be scored
    preds = RF.predict_proba(X_to_score)[:, 1]

    return preds