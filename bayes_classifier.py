from __future__ import print_function, division
from datetime import datetime
from sklearn.utils import shuffle
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

import numpy as np
import pandas as pd


cd = '~/Documents/machine_learning_examples/mnist_csv/'

show_scores = True
show_times = True

limit = 10000
proportion_train = 0.75


class BayesClassifier(object):
    def fit(self, X, Y, smoothing=10e-3):
        D = X.shape[1]
        self.gaussians = dict()
        self.priors = dict()

        labels = set(Y)

        for c in labels:
            xc = X[Y == c]

            self.gaussians[c] = {
                'mean': xc.mean(axis=0),
                'cov': np.cov(xc.T) + np.eye(D) * smoothing
            }
            
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        
        P = np.zeros((N, K))

        for c, g in self.gaussians.iteritems():
            mu, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mu, cov=cov) + np.log(self.priors[c])

        return np.argmax(P, axis=1)
                
    def score(self, X, Y):
        P = self.predict(X)
        
        return np.mean(P == Y)

    
def get_data(cd, limit=None):
    print('Reading in and transforming data...')
    
    df = pd.read_csv(cd + 'train.csv')
    data = df.as_matrix()
    
    np.random.shuffle(data)
    
    X = data[:, 1:]
    Y = data[:, 0]
    
    if limit is not None:
        X, Y = X[:limit], Y[:limit]

    print('...Finished\n')
        
    return X, Y


if __name__ == '__main__':

    t0 = datetime.now()

    X, Y = get_data(cd, limit=limit)

    Ntrain = int(proportion_train * len(Y))

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = BayesClassifier()

    t1 = datetime.now()

    print('Fitting...')
    model.fit(Xtrain, Ytrain)

    t2 = datetime.now()

    print('Scoring train...')
    sTrain = model.score(Xtrain, Ytrain)

    t3 = datetime.now()

    print('Scoring test...\n')
    sTest = model.score(Xtest, Ytest)

    t4 = datetime.now()

    if show_scores:
        print('Train Accuracy:', 100 * np.round(sTrain, 3))
        print('Test Accuracy:', 100 * np.round(sTest, 3), '%\n')

    if show_times:
        print('Loading Time:', (t1 - t0))
        print('Fitting Time:', (t2 - t1))
        print('Scoring Time (Train):', (t3 - t2))
        print('Scoring Time (Test):', (t4 - t3))
        

    
