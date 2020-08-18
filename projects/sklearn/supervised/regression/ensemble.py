from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)
print("Max:", np.max(y))
print("Min:", np.min(y))
print("Mean:", np.mean(y))

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

em = ExtraTreesRegressor()
em.fit(X_train, y_train)
y_predict = em.predict(X_test)
print("\n\nExtraTreesRegressor:")
print("r2 score:", r2_score(y_test, y_predict))
print("mse:", mean_squared_error(y_test, y_predict))
print("mae:", mean_absolute_error(y_test, y_predict))

em = GradientBoostingRegressor()
em.fit(X_train, y_train)
y_predict = em.predict(X_test)
print("\n\nGradientBoostingRegressor:")
print("r2 score:", r2_score(y_test, y_predict))
print("mse:", mean_squared_error(y_test, y_predict))
print("mae:", mean_absolute_error(y_test, y_predict))

em = RandomForestRegressor()
em.fit(X_train, y_train)
y_predict = em.predict(X_test)
print("\n\nRandomForestRegressor:")
print("r2 score:", r2_score(y_test, y_predict))
print("mse:", mean_squared_error(y_test, y_predict))
print("mae:", mean_absolute_error(y_test, y_predict))
