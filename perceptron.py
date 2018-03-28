from __future__ import print_function

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)

cd = '~/Documents/machine_learning_examples/mnist_csv/'

show_graph = True
show_messages = True
show_weights = True
show_scores = True
show_times = True
show_costs = True

load_limit = 1000
proportion_train = 0.75
learning_rate = 0.1

epochs = 5000
alert_at = 1000


class Perceptron(object):
    def __init__(self, learning_rate=learning_rate, epochs=epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.costs = []

    def fit(self, X, T):

        N, D = X.shape

        self.w = np.random.randn(D)
        self.b = 0

        for i in xrange(self.epochs):

            if i % alert_at == 0:
                print(i)

            errors = self.find_errors(X, T)
            
            if errors is None:
                break

            for error in errors:
                x = error[0]
                t = error[1]
                 
                self.w += self.learning_rate * t * x
                self.b += self.learning_rate * t

        cost = len(errors) / N
        self.costs.append(cost)

        print(self.costs)

    def find_errors(self, X, T):

        errors = []
            
        for i, (x, t) in enumerate(zip(X, T)):
            
            p = self.predict(x)

            if p != t:
                errors.append([x, t])

        return errors

    def predict(self, X):

        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, T):

        P = self.predict(X)

        return np.mean(P == T)


def get_data():
    
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    T = np.sign(X.dot(w) + b)
    
    return X, T


def get_mnist(cd, limit=None):
    
    print('Reading in and transforming data...')
    
    df = pd.read_csv(cd + 'train.csv')
    data = df.as_matrix()
    
    np.random.shuffle(data)
    
    X = data[:, 1:]
    T = data[:, 0]
    
    if limit is not None:
        X, T = X[:limit], T[:limit]

    X = X / 255.0

    print('...Finished\n')
        
    return X, T


def get_xor():
    X = np.zeros((200, 2))
    
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)

    T = np.array([0]*100 + [1]*100)
    
    return X, T


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    T = np.array([0]*(N//2) + [1]*(N//2))
    
    return X, T


if __name__ == '__main__':

    t0 = datetime.now()

    #X, T = get_mnist(cd, load_limit)
    #X, T = get_xor()
    #X, T = get_donut()

    X, T = get_data()

    if show_graph:
        plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
        plt.show()

    t1 = datetime.now()

    N = X.shape[0]
    Ntrain = int(N * proportion_train)

    Xtrain, Ttrain = X[:Ntrain], T[:Ntrain]
    Xtest, Ttest = X[Ntrain:], T[Ntrain:]

    t2 = datetime.now()

    model = Perceptron()


    if show_messages:
        print('Fitting...')
    
    model.fit(Xtrain, Ttrain)

    t3 = datetime.now()

    if show_messages:
        print('Finished fitting...\n\nScoring train...')

    train_acc = model.score(Xtrain, Ttrain)

    if show_messages:
        print('Scoring test...')
            
    test_acc = model.score(Xtest, Ttest)

    if show_messages:
       print('Finished scoring...\n')

    if show_weights:
        print('Final w:', model.w)
        print('Final b:', model.b)

    if show_scores:
        print('Train Accuracy:', train_acc)
        print('Test Accuracy:', test_acc, '\n')

    if show_costs:
        plt.plot(model.costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        
        plt.show()
        
