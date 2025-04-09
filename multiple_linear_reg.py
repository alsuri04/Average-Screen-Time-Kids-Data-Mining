# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Training the Multiple Linear Regression model on the Training set



# Building the optimal model using Backward Elimination
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled=sc_X.transform(X_test)
y_train_scaled = sc_Y.fit_transform(y_train.reshape(len(y_train), 1)).flatten()
y_test_scaled = sc_Y.transform(y_test.reshape(len(y_test), 1)).flatten()

# Backward Elimination
X_train_scaled = sm.add_constant(X_train_scaled).astype(np.float64)
X_opt = X_train_scaled[:, [0, 1, 2, 3, 4]]
regressor_opt = sm.OLS(endog=y_train_scaled, exog=X_opt).fit()
regressor_opt.summary()


y_pred=regressor_opt.predict(X_test_scaled)
y_pred_unscaled = sc_Y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Evaluating the Model Performance (Preferred)
## Universal for all regression models
## Most convincing when used on test sets
## Applicable to both training and test sets
from sklearn.metrics import r2_score
r2 = r2_score(y_test_scaled, y_pred)
print(r2)
r2_adjusted = 1 - (1 - r2) * (len(y_test_scaled) - 1) / (len(y_test_scaled) - 6 - 1)
print('adjusted ', r2_adjusted)

X_past_score=X_test[:,2]
plt.scatter(X_past_score, y_test,color = 'red')
plt.scatter(X_test[:, 2], y_pred_unscaled, color='blue', marker='x', label='Predicted')

plt.title('SVR Regression')
plt.xlabel('Past Test Scores')
plt.ylabel('Current Performance Index')
plt.show()

