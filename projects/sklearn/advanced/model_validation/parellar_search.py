from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import time
news = fetch_20newsgroups(subset="all")
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33
)

clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
    ('svc', SVC())
])

parameters = {
    'svc__gamma': np.logspace(-2, 1, 4),
    'svc__C': np.logspace(-1, 1, 3)
}

gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
t1 = time.time()
gs.fit(X_train, y_train)
t2 = time.time()
print("Used time: ", t2-t1)
print("Best parameters:", gs.best_params_)
print("Best Score:", gs.best_score_)
