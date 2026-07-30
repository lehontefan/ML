import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mylib.FFN import FFNClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]
X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
ffn = FFNClassifier(X_train.shape[1], len(np.unique(y)), 1, X_train.shape[1] * 2, learning_rate=0.5, iterations=100)
ffn.fit(X_train, y_train)
y_pred = np.argmax(ffn.predict(X_test), axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")