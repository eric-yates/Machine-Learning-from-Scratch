import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

class BaggedTreeRegressor:

        def __init__(self, B, Nb):
            self.B = B
            self.Nb = Nb

        def fit(self, X, T):
            """
            Xb: Bootstrapped samples of X
            Tb: Bootstrapped samples of T
            """

            self.models = []

            for b in range(self.B):
                
                idx = np.random.choice(X.shape[0], self.Nb)
                Xb = X[idx]
                Tb = T[idx]
                
                model = DecisionTreeRegressor()
                model.fit(Xb, Tb)
                
                self.models.append(model)

        def predict(self, X):

            self.Y = []

            for model in self.models:
                self.Y.append(model.predict(X))

            self.P = np.mean(self.Y, axis=0)

        def score(self, X, T):

            self.predict(X)

            self.acc = r2(T, self.P)

            return self.acc
        

if __name__ == '__main__':
    # The proportion of train-to-test data
    prop_train = 0.8

    # Number of bootstrap models
    B = 500
    
    # Number of samples with replacement per bootstrap
    Nb = 30
    
    # create the data
    Nt = 100
    X = np.linspace(0, 2*np.pi, Nt)
    signal = np.sin(X)
    noise = np.random.normal(0, 0.25, 100)
    T = signal + noise
    X = X.reshape(Nt, 1)

    Xtrain, Xtest, Ttrain, Ttest = train_test_split(X, T, random_state=7,
                                                     test_size=1-prop_train)

    # try a lone decision tree
    model = DecisionTreeRegressor()
    model.fit(Xtrain, Ttrain)
    prediction = model.predict(X)

    btr = BaggedTreeRegressor(B=B, Nb=Nb)
    
    btr.fit(Xtrain, Ttrain)

    btr.predict(X)

    # plot the lone decision tree's predictions
    plt.plot(X, signal, c='r', label='Signal')
    plt.plot(X, prediction, c='b', label='Single decision tree')
    plt.scatter(X, T, c='black', label='Noisy signal', s=10)
    plt.title('Single Decision Tree')
    plt.legend()
    plt.show()

    # plot the bagged decision tree's predictions
    plt.plot(X, signal, c='r', label='Signal')
    plt.plot(X, btr.P, c='b', label='Bagged decision tree')
    plt.scatter(X, T, c='black', label='Noisy signal', s=10)
    plt.title('Bagged Decision Tree')
    plt.legend()
    plt.show()

    train_score = btr.score(Xtrain, Ttrain)
    test_score = btr.score(Xtest, Ttest)

    print('Train score for 1 tree:', model.score(Xtrain, Ttrain))
    print(' Test score for 1 tree:', model.score(Xtest, Ttest))
    print('Train Score for bagged:', train_score)
    print(' Test Score for bagged:', test_score)




