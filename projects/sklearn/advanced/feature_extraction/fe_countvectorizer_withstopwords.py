#! /bin/bash python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(
    news.data,
    news.target,
    test_size=0.25,
    random_state=33)

count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_count_train, y_train)
print('Accuracy of classifying 20newsgroup using Naive Bayes', mnb.score(X_count_test, y_test))

y_count_predict = mnb.predict(X_count_test)
print(classification_report(y_test, y_count_predict, target_names=news.target_names))
