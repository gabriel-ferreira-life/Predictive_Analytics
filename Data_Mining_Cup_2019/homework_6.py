import boto3
import pandas as pd; pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
import precision_recall_cutoff
import Feature_Importance_Funs

# reading updated train dataset
train = pd.read_csv('train_dataset.csv')
train = train.dropna()

## Engineering features using the strong heredity principle
train['heredity_1'] = train['interaction_1'] * train['trustLevel']

train['heredity_2'] = train['interaction_1'] * train['Labels']

train['heredity_3'] = train['trustLevel'] * train['Labels']

# Variable created in the last feature engineering section
train['interaction_9'] = np.where(train['heredity_1'] > 0.5, 1, 0)

logit_list = list()
RF_list = list()
Ada_list = list()

for i in range(0,100):
    
    # Defining input and target variables
    X = train.drop(['fraud'], axis = 1)
    Y = train['fraud']

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)

    # Standardizing the dataset
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)
    
    # Running RFE with LogisticRegression
    logit_rfe = RFE(estimator = LogisticRegression(), n_features_to_select = 5).fit(X_train, Y_train)# Extracting features that got 
    
    # Extracting features that got slected
    logit_list.append(X_train.columns[logit_rfe.support_])
    
    # Running RFE with random forest
    RF_rfe = RFE(estimator = RandomForestClassifier(n_estimators = 500, max_depth = 3), n_features_to_select = 5).fit(X_train, Y_train)

    # Extracting features that got slected
    RF_list.append(X_train.columns[RF_rfe.support_])
    
    # Running RFE with AdaBoost
    Ada_rfe = RFE(estimator = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3), n_estimators = 500, learning_rate = 0.01), n_features_to_select = 5).fit(X_train, Y_train)

    # Extracting features that got slected
    Ada_list.append(X_train.columns[Ada_rfe.support_])

    if i == 0:
        print('Iteraction :', i, end=' ')
    else:
        print(i, end=' ')

        
logit = pd.DataFrame(logit_list)
RF = pd.DataFrame(RF_list)
ada = pd.DataFrame(Ada_list)

logit.to_csv('logit_list.csv', index = False)
RF.to_csv('RF_list.csv', index = False)
ada.to_csv('ada_list.csv', index = False)