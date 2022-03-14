import pandas as pd; pd.set_option('display.max_columns', 50)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import precision_recall_cutoff # Calling .py function
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['interaction_1'] = np.where((train['EDUCATION'] == 1) | (train['EDUCATION'] == 2), 1, 0)
train['interaction_2'] = train['PAY_0']*train['PAY_2']*train['PAY_3']*train['PAY_4']*train['PAY_5']
train['interaction_3'] = (train['PAY_0']+train['PAY_2']+train['PAY_3']+train['PAY_4']+train['PAY_5'])/5
train['interaction_4'] = round((train['BILL_AMT1']+train['BILL_AMT2']+train['BILL_AMT3']+train['BILL_AMT4']+train['BILL_AMT5']+train['BILL_AMT6'])/6, 2)
train['interaction_5'] = round((train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']+train['PAY_AMT5']+train['PAY_AMT6'])/6, 2)
train['interaction_6'] = train['interaction_4'] - train['interaction_5']
train['interaction_7'] = train['BILL_AMT1']+train['BILL_AMT2']+train['BILL_AMT3']+train['BILL_AMT4']+train['BILL_AMT5']+train['BILL_AMT6']
train['interaction_8'] = train['PAY_AMT1']+train['PAY_AMT2']+train['PAY_AMT3']+train['PAY_AMT4']+train['PAY_AMT5']+train['PAY_AMT6']
train['interaction_9'] = train['interaction_7'] - train['interaction_8']

train['heredity_1'] = np.where((train['PAY_0'] <= 1.5) & (train['interaction_3'] <= 0.5), 1, 0)
train['heredity_2'] = np.where(train['PAY_0'] > 1.5, 1, 0)

RF_list = list()

for i in range(0,100):
    
    ## Creating ID columns
    train['ID'] = list(range(1, train.shape[0] + 1))

    ## Splitting the data into train and test
    training = train.groupby('default payment next month', group_keys = False).apply(lambda x: x.sample(frac = 0.8))
    testing = train[~np.isin(train['ID'], training['ID'])]

    ## Dropping ID
    training = training.drop(columns = 'ID', axis = 1)
    testing = testing.drop(columns = 'ID', axis = 1)

    X_train = training.drop('default payment next month', axis = 1)
    Y_train = training['default payment next month']
    X_test = testing.drop('default payment next month', axis = 1)
    Y_test = testing['default payment next month']
    
    # Running RFECV with random forest
    RF_rfecv = RFECV(estimator = RandomForestClassifier(n_estimators = 500, max_depth = 3), step = 1, min_features_to_select = 2, cv = 3).fit(X_train, Y_train)

    # Extracting features that got slected
    RF_list.append(X_train.columns[RF_rfecv.support_])

RF = pd.DataFrame(RF_list)
RF.to_csv('RF_list.csv', index = False)