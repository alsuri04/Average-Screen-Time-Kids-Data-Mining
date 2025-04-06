# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:56:59 2025

@author: thaip
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Student_Performance.csv')

X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])],
                       remainder='passthrough')
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X, y, test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()

X_train_svr = sc_X.fit_transform(X_train_raw)
X_test_svr = sc_X.transform(X_test_raw)
Y_train_svr = sc_Y.fit_transform(Y_train_raw.reshape(-1, 1)).flatten()
Y_test_svr = sc_Y.transform(Y_test_raw.reshape(-1, 1)).flatten()
# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_svr, Y_train_svr)

y_pred_svr=regressor.predict(X_test_svr)
y_pred_unscaled = sc_Y.inverse_transform(y_pred_svr.reshape(-1, 1)).flatten()

# Evaluating the Model Performance (Preferred)
## Universal for all regression models
## Most convincing when used on test sets
## Applicable to both training and test sets
from sklearn.metrics import r2_score
r2 = r2_score(Y_test_svr, y_pred_svr)
print(r2)
r2_adjusted = 1 - (1 - r2) * (len(Y_test_svr) - 1) / (len(Y_test_svr) - 6 - 1)
print('adjusted ', r2_adjusted)

X_past_score=X_test_raw[:,2]
plt.scatter(X_past_score, Y_test_raw,color = 'red')
plt.scatter(X_test_raw[:, 2], y_pred_unscaled, color='blue', marker='x', label='Predicted')

plt.title('SVR Regression')
plt.xlabel('Past Test Scores')
plt.ylabel('Current Performance Index')
plt.show()
