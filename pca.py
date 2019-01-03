import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import matplotlib
matplotlib.use('Tkagg')
from matplotlib import pyplot as plt

class PCA():

    def fit(self, X, T):
        """
        Transforms input data X into a reduced dataset via principal components
        analysis (PCA). Results in a matrix sorted by the information content
        of each dimension. Dimensions carrying the most information (ie.
        statistical variance) appear as the first columns. Dimensions carrying
        the least information appear as the last columns.
        """

        self.X = X
        self.T = T

        # Calculate the covariance of X
        covX = np.cov(X, rowvar=False)

        # Calculate the eigenvalues (W) and eigenvectors (V) of the covariance
        W, V = np.linalg.eigh(covX)

        # Sort eigenvalues/eigenvectors in descending order
        idx = np.argsort(-W)
        self.W = W[idx]
        self.V = V[:, idx]

        # Find the reduced data
        self.Z = X.dot(self.V)

    def plot_reduced(self):
        """
        Plots the two dimensions with the most information from the PCA
        analysis. Allows for data visualization of high dimensional data
        through dimensionality reduction.
        """

        x = self.Z[:, 0]    # Primary dimension (most information)
        y = self.Z[:, 1]    # Secondary dimension (second most information)

        labels = list(set(self.T))
        colors = [plt.cm.jet(float(i)/max(labels)) for i in labels]

        for i, u in enumerate(labels):
            xi = [x[j] for j in range(len(x)) if self.T[j] == u]
            yi = [y[j] for j in range(len(y)) if self.T[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(u), alpha=0.3)

        plt.title('Reduced MNIST dataset')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def plot_variance(self):
        plt.plot(self.W)
        plt.title('Variance of each component')
        plt.show()

    def plot_cumulative_variance(self):
        plt.plot(np.cumsum(self.W))
        plt.title('Cumulative variance')
        plt.show()


def get_data():
    """
    Dataset obtained from the Kaggle MNIST challenge under the Creative Commons
    license. Contains 42,000 images of size 28x28 flattened to a vector of
    length 784.

        Column 0        Labels
        Columns 1-785   Pixel values from 0...255
    """

    mnist = pd.read_csv('data/mnist.csv').values.astype(np.float32)
    mnist = shuffle(mnist)

    Xtrain = mnist[:-1000,1:] / 255
    Ttrain = mnist[:-1000,0].astype(np.int32)
    Xtest  = mnist[-1000:,1:] / 255
    Ttest  = mnist[-1000:,0].astype(np.int32)

    return Xtrain, Ttrain, Xtest, Ttest

if __name__ == '__main__':

    Xtrain, Ttrain, Xtest, Ttest = get_data()

    pca = PCA()
    pca.fit(Xtrain, Ttrain)
    pca.plot_reduced()
    pca.plot_variance()
    pca.plot_cumulative_variance()
