#! /usr/bin/env python
import math
import numpy as np
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def load_training_set():
    X = []
    y = []
    with open('train.txt') as f:
        for line in f:
            data = line.strip().split(',')
            label = int(float(data[-1]))
            sample = [float(x) for x in data[:-1]]
            X.append(sample)
            y.append(label)
    return np.array(X), np.array(y)


def load_test_set():
    X = []
    with open('test.txt') as f:
        for line in f:
            data = line.strip().split(',')
            sample = [float(x) for x in data]
            X.append(sample)
    return np.array(X)


def find_best_model(X, y):
    models = [
        (DecisionTreeClassifier, [
            {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1, 2,3,4], 'min_samples_split': [1,2,3,4] }
        ]),
        (RandomForestClassifier, [
            {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1, 2,3,4], 'min_samples_split': [1,2,3,4] }
        ]),
        (GradientBoostingClassifier, [
            {'loss': ['deviance', 'exponential'], 'max_depth': [None, 1, 2,3,4], 'min_samples_split': [1,2,3,4] }
        ]),
        (SVC, [
#            {'C': [math.pow(10, power) for power in xrange(-4, 5)], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [math.pow(2, power) for power in xrange(-4, 5)]}
            {'C': [math.pow(10, power) for power in xrange(-4, 5)], 'kernel': ['rbf', 'sigmoid'], 'gamma': [math.pow(2, power) for power in xrange(-4, 5)]}
        ]),
        (LinearDiscriminantAnalysis, [
            {'solver': ['svd']},
            {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}
        ]),
    ]

    best_score = 0
    best_gs = None
    for estimatorclass, param_grid in models:
        print "Train %s" % estimatorclass.__name__
        gs = grid_search.GridSearchCV(estimator=estimatorclass(), param_grid=param_grid, cv=5)
        gs.fit(X, y)
        print "best params = %s\nbest score = %f\n" % (gs.best_params_, gs.best_score_)
        if gs.best_score_> best_score:
            best_score = gs.best_score_
            best_gs = gs
    print "Best estimator: %s with params %s, best score: %f" % (type(best_gs.best_estimator_).__name__, best_gs.best_params_, best_gs.best_score_)
    return best_gs


def main():
    X_train, y_train = load_training_set()
    # normalize each feature
    X_train = preprocessing.scale(X_train)

    gs = find_best_model(X_train, y_train)

    X_test = load_test_set()
    # normalize each feature
    X_test = preprocessing.scale(X_test)

    y_test = gs.predict(X_test)
    print y_test
    with open('label.txt', 'w') as f:
        for y in y_test:
            f.write("%d\n" % y)


if __name__ == '__main__':
    main()
