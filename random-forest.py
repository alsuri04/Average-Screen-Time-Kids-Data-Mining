#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:30:52 2025

@author: marybethwalsh
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
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


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500)  
regressor.fit(X_train_raw, Y_train_raw)



y_pred_rf=regressor.predict(X_test_raw)

# Evaluating the Model Performance (Preferred)
## Universal for all regression models
## Most convincing when used on test sets
## Applicable to both training and test sets
from sklearn.metrics import r2_score
r2 = r2_score(Y_test_raw, y_pred_rf)
print(r2)
r2_adjusted = 1 - (1 - r2) * (len(Y_test_raw) - 1) / (len(Y_test_raw) - 5 - 1)
print('adjusted ', r2_adjusted)


# Visualizing the Random Forest Regression results (for higher resolution and
X_past_score=X_test_raw[:,2]
plt.scatter(X_past_score, Y_test_raw,color = 'red')
plt.scatter(X_test_raw[:, 2], y_pred_rf, color='blue', marker='x', label='Predicted')

plt.title('Random Forest Regression')
plt.xlabel('Past Test Scores')
plt.ylabel('Current Performance Index')
plt.show()