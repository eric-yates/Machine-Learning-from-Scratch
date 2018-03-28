from __future__ import print_function

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)

cd = '~/Documents/machine_learning_examples/mnist_csv/'

show_messages = False
show_scores = False
show_times = False
show_IG = False

limit = 1000
proportion_train = 0.75
max_depth = 2

iterations = 10000

class DecisionTree(object):
    def __init__(self):
        self.max_depth = max_depth

    def fit(self, X, T):
        self.root_node = TreeNode(current_depth=1)
        self.root_node.fit(X, T)

    def predict(self, X):

        P=[]

        for x in X:
            P.append(self.root_node.predict(x))

        return P

    def score(self, X, T):

        P = self.predict(X)
        
        return np.mean(P == T)


class TreeNode(object):
    def __init__(self, current_depth):        
        self.current_depth = current_depth + 1
        self.max_depth = max_depth

        self.left_node = None
        self.right_node = None
        self.left_P = None
        self.right_P = None

    def fit(self, X, T):

        self.X = X
        self.T = T

        self.best_IG = 0
        self.condition = None
        self.best_i = None

        for i, col in enumerate(self.X.T):

            sort_idx = np.argsort(col)
            x = col[sort_idx]
            t = self.T[sort_idx]
            
            splits = self.find_splits(t)

            if not splits:
                T_left = t

            else:               
                for split in splits:

                    T_left = t[:split]
                    T_right = t[split:]

                    N_left = len(T_left)
                    N_right = len(T_right)

                    N_total = float(N_left + N_right)
                    
                    info_gain = (self.entropy(self.T)
                                 - N_left / N_total * self.entropy(T_left)
                                 - N_right / N_total * self.entropy(T_right))

                    if info_gain > self.best_IG:
                        self.best_IG = info_gain
                        self.condition = x[split]
                        self.best_i = i

 #       X_left, X_right, T_left, T_right = self.get_splits()

        if self.current_depth > self.max_depth or self.best_IG == 0:

            self.left_P, self.right_P = self.make_leaves(T_left)

            return

        if show_IG:
            print('Column Number:', self.best_i)
            print('Best Condition:', self.condition)
            print('Best Information Gain:', self.best_IG, '\n')

        X_left, X_right, T_left, T_right = self.get_splits()

        ## print('Fitting left node', self.current_depth)
        self.left_node = TreeNode(current_depth=self.current_depth)
        self.left_node.fit(X_left, T_left)

        ## print('Fitting right node', self.current_depth)
        self.right_node = TreeNode(current_depth = self.current_depth)
        self.right_node.fit(X_right, T_right)

        return
            
    def predict(self, x):

        if self.best_i is None:
            return 1

        elif x[self.best_i] <= self.condition:
            if self.left_node:
                return self.left_node.predict(x)
            else:
                return self.left_P
                
        else:
            if self.right_node:
                return self.right_node.predict(x)
            else:
                return self.right_P

    def find_splits(self, t):

        splits = []
        prev_t = t[0]

        for i, current_t in enumerate(t):
            if current_t != prev_t:
                splits.append(i)
            prev_t = current_t

        return splits

    def calc_prop(self, T):
        
        n1 = T[T == 1].shape[0]
        N = float(T.shape[0])

        return n1 / N
        
    def entropy(self, T):

        p = self.calc_prop(T)

        if p == 0:
            p = 0.00001

        elif p == 1:
            p = 0.99999

        return -p * np.log(p) - (1 - p) * np.log(1 - p)

    def get_splits(self):

        left_idx = np.where(self.X.T[self.best_i] <= self.condition)
        right_idx = np.where(self.X.T[self.best_i] > self.condition)

        X_left = self.X[left_idx]
        X_right = self.X[right_idx]
        
        T_left = self.T[left_idx]
        T_right = self.T[right_idx]

        return X_left, X_right, T_left, T_right

    def make_leaves(self, T):

        p = self.calc_prop(T)

        if p >= 0.5:
            return 1, 0

        else:
            return 0, 1
        

def get_data(cd, limit=None):
    
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

    train_accs = []
    test_accs = []

    for i in xrange(iterations):
        t0 = datetime.now()
        
    #    X, T = get_data(cd, limit=limit)
        X, T = get_donut()

        t1 = datetime.now()

        idx = np.logical_or(T == 0, T == 1)
        X = X[idx]
        T = T[idx]

        X, T = shuffle(X, T)

        t2 = datetime.now()

        N = X.shape[0]

        Ntrain = int(N * proportion_train)

        Xtrain, Ttrain = X[:Ntrain], T[:Ntrain]
        Xtest, Ttest = X[Ntrain:], T[Ntrain:]

        model = DecisionTree()

        t3 = datetime.now()

        if show_messages:
           print('Fitting...')

        model.fit(Xtrain, Ttrain)

        t4 = datetime.now()

        if show_messages:
            print('Finished fitting...\n')
            print('Scoring train...')

        train_acc = model.score(Xtrain, Ttrain)

        t5 = datetime.now()
        
        if show_messages:
            print('Scoring test...')

        test_acc = model.score(Xtest, Ttest)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        t6 = datetime.now()

        if show_messages:
            print('Finished scoring...\n')

        if show_scores:
            print('Train Accuracy:', train_acc)
            print('Test Accuracy:', test_acc)

        if show_times:
            print('Loading Time:', t1 - t0)
            print('Pruning Time:', t2 - t1)
            print('Splitting Time:', t3 - t2)
            print('Fitting Time:', t4 - t3)
            print('Train Scoring Time:', t5 - t4)
            print('Test Scoring Time:', t6 - t5)

    plt.hist(train_accs, np.linspace(0, 1, 100))
    plt.show()

    plt.hist(test_accs, np.linspace(0, 1, 100))
    plt.show()

