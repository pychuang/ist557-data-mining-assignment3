#! /usr/bin/env python
import math
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def iter_param_grid(param_grid):
    def iter_internal(param_row, result, keys):
        if not keys:
            yield result
        else:
            k = keys[0]
            for v in param_row[k]:
                result[k] = v
                for r in iter_internal(param_row, result, keys[1:]):
                    yield r

    for param_row in param_grid:
        for result in iter_internal(param_row, {}, param_row.keys()):
            yield result


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
            {'C': [math.pow(10, power) for power in xrange(-4, 5)], 'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': ['auto'] + [math.pow(2, power) for power in xrange(-4, 5)]}
        ]),
        (LinearDiscriminantAnalysis, [
            {'solver': ['svd']},
            {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto']}
        ]),
        (LogisticRegression, [
            {'solver': ['newton-cg','lbfgs', 'sag'], 'C': [math.pow(10, power) for power in xrange(-4, 5)],},
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [math.pow(10, power) for power in xrange(-4, 5)],},
        ]),
    ]

    best_score = 0
    best_model = None
    best_params = None

    indices = []
    # 5-fold
    for train_index, valid_index in cross_validation.KFold(n=len(X), n_folds=5, shuffle=True):
        indices.append((train_index, valid_index))

    for estimatorclass, param_grid in models:
        for params in iter_param_grid(param_grid):
            scores = []
            for train_index, valid_index in indices:
                X_train = X[train_index]
                y_train = y[train_index]
                X_valid = X[valid_index]
                y_valid = y[valid_index]

                estimator = estimatorclass(**params).fit(X_train, y_train)
                score = estimator.score(X_valid, y_valid)
                scores.append(score)

            avg_score = sum(scores) / len(scores)
            print "%s with params %s:\t%f" % (estimatorclass.__name__, params, avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_model = estimator
                best_params = params

    print
    print "Best model: %s with params %s, best score: %f" % (type(best_model).__name__, best_params, best_score)
    return best_model


def main():
    X_train, y_train = load_training_set()
    # normalize each feature
    X_train = preprocessing.scale(X_train)

    model = find_best_model(X_train, y_train)

    X_test = load_test_set()
    # normalize each feature
    X_test = preprocessing.scale(X_test)

    y_test = model.predict(X_test)
    print
    print y_test
    with open('label.txt', 'w') as f:
        for y in y_test:
            f.write("%d\n" % y)


if __name__ == '__main__':
    main()
