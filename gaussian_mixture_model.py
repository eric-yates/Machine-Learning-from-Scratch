import numpy as np
from scipy.stats import multivariate_normal as mvn

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class GMM():
    def __init__(self, K, epochs=30):
        """
        Initializes parameters for the GMM model.
        """

        self.K = K
        self.epochs = 30

    def fit(self, X, smoothing=0.01):
        """
        Fits to the data by iteratively calculating the responsibilities for
        each data point and updating the model parameters. Exits early if the
        loss is less than 0.1 smaller than the previous loss.
        """

        N, D = X.shape
        costs = []

        ### Matrix initialization
        R = np.zeros((N, self.K))            # Responsibilities
        M = np.zeros((self.K, D))            # Means
        C = np.zeros((self.K, D, D))         # Covariances
        P = np.ones(self.K) / self.K         # Probabilities

        for k in range(self.K):
            M[k] = X[np.random.choice(N)]
            C[k] = np.eye(D)

        ### Fitting to the data X
        for i in range(self.epochs):

            ### Calculate responsibilities for each Gaussian k
            weighted_pdfs = np.zeros((N, self.K))
            for k in range(self.K):
                weighted_pdfs[:, k] = P[k] * mvn.pdf(X, M[k], C[k])
            R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

            ### Update parameters for each Gaussian k
            for k in range(self.K):
                Nk = R[:, k].sum()
                P[k] = Nk / N
                M[k] = R[:, k].dot(X) / Nk

                delta = X - M[k]
                Rdelta = np.expand_dims(R[:,k], -1) * delta
                C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D) * smoothing

            cost = np.log(weighted_pdfs.sum(axis=1)).sum()
            costs.append(cost)

            if i > 0 and np.abs(cost - costs[-2]) < 0.1:
                print('Early exit at i = {}! Small change in cost.'.format(i))
                break

        plt.plot(costs)
        plt.title('Costs for GMM')
        plt.show()

        random_colors = np.random.random((self.K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.title('Cluster assignments')
        plt.show()


def get_data(noise_factor=1, sep=4):
    """
    Generates 3 clusters of data of 200 data points each. Can tweak the
    distribution of data with the noise_factor (ie. how spread apart data points
    are) and the sep (ie. how far separated actual cluster centers are).
    """

    D = 2 # Two-dimensional data to easily visualize

    m1 = np.array([0, 0])
    m2 = np.array([sep, sep])
    m3 = np.array([0, sep])

    N = 600
    X = np.zeros((N, D))
    X[:100, :] = noise_factor * np.random.randn(100, D)*2 + m1
    X[100:400, :] = noise_factor * np.random.randn(300, D) + m2
    X[400:, :] = noise_factor * np.random.randn(200, D)*0.5 + m3

    return X


def main():
    """
    Generates a dataset of three Gaussian-distributed clusters, fits to a GMM
    model with K = 3, and plots the results.
    """

    X = get_data()

    model = GMM(K=3)
    model.fit(X)

if __name__ == '__main__':
    main()
