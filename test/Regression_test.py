import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mylib.LinearRegression import LinearRegression
from mylib.LogisticRegression import LogisticRegression

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]
X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]
X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = np.argmax(logistic_regression.predict(X_test), axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")