import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

column_names = [
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'
]

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)

data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

t1 = time.time()
sgdc = SGDClassifier()
sgdc.fit(X_train, y_train)
score = sgdc.score(X_test, y_test)
print("Accuracy: ", score)
t2 = time.time()
print("Use time: ", t2-t1)

from sklearn.metrics import classification_report
predict = sgdc.predict(X_test)
report = classification_report(y_test, predict, target_names = ['Benign', 'Malignant'])
print(report)

