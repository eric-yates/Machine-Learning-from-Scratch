from __future__ import print_function, division
from future.utils import iteritems
from datetime import datetime
from sortedcontainers import SortedList
from sklearn.utils import shuffle
from util import get_data, get_xor, get_donut

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cd = '~/Documents/machine_learning_examples/mnist_csv/'

show_scores = True
show_times = True

limit = None
proportion_train = 0.75


class KNN(object):
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))

        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                distance = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((distance, self.y[j]))
                elif distance < sl[-1][0]:
                    del sl[-1]
                    sl.add((distance, self.y[j]))

            votes = {}

            for _, vote in sl:
                votes[vote] = votes.get(vote, 0) + 1

            max_count = 0
            max_count_class = -1

            for vote, count in votes.iteritems():
                if count > max_count:
                    max_count = count
                    max_count_class = vote

            y[i] = max_count_class

        return y

    def score(self, X, Y):
        P = self.predict(X)

        return np.mean(P == Y)


def get_mnist(cd, limit=None):

    X = pd.read_csv(cd + 'Xtrain.txt').as_matrix()
    Y = pd.read_csv(cd + 'label_train.txt').as_matrix()

    X, Y = shuffle(X, Y)

    y = np.zeros(len(Y))

    for i, yt in enumerate(Y):
        y[i] = yt[0]

    if limit is not None:
        return X[:limit], y[:limit]

    else:
        return X, y


if __name__ == '__main__':
    
    #X, Y = get_mnist(cd, limit=limit)

    X, Y = get_donut()

    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()

    Ntrain = int(proportion_train * len(X))

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    train_scores = []
    test_scores = []

    ks = (1, 2, 3, 4, 5)

    for k in ks:
        print('k =', k)
                
        model = KNN(k)
        t0 = datetime.now()

        model.train(Xtrain, Ytrain)

        t1 = datetime.now()

        train_score = model.score(Xtrain, Ytrain)
        train_scores.append(train_score)

        t2 = datetime.now()

        test_score = model.score(Xtest, Ytest)
        test_scores.append(test_score)

        t3 = datetime.now()

        if show_scores:
            print('Training Accuracy:', train_score)
            print('Test Accuracy:', test_score)

        if show_times:
            print('Training Time:', (t1 - t0))
            print('Prediction Time (Train):', (t2 - t1))
            print('Prediction Time (Test):', (t3 - t2), '\n')
        
    plt.plot(ks, train_scores, label='Train Scores')
    plt.plot(ks, test_scores, label='Test Scores')
    plt.legend()
    plt.show()
    
