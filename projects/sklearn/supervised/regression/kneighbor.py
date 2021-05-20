from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)
print("Max:", np.max(y))
print("Min:", np.min(y))
print("Mean:", np.mean(y))

ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_y_predict = uni_knr.predict(X_test)

print("Accuracy of uniform K nearest neighbor Regression:", uni_knr.score(X_test, y_test))
print("uniform mse:", mean_squared_error(y_test, uni_y_predict))
print("uniform ae:", mean_absolute_error(y_test, uni_y_predict))

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_y_predict = dis_knr.predict(X_test)

print("Accuracy of distance K nearest neighbor Regression:", dis_knr.score(X_test, y_test))
print("dis_knr mse:", mean_squared_error(y_test, dis_y_predict))
print("dis_knr ae:", mean_absolute_error(y_test, dis_y_predict))

