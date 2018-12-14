import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class SoftKMeansClustering():
    def __init__(self, K, beta):
        self.K = K
        self.beta = beta

    def fit(self, X):
        self.centers = X[np.random.choice(X.shape[0], self.K, replace=False), :]

        N = X.shape[0]
        self.resps = np.zeros((N, self.K))
        self.costs = []

        while True:
            # Calculate cluster responsibilities by proximity
            self.calc_responsibilities(X)

            # Recalculate centers by aggregation
            for k in range(self.K):
                cols = [self.resps[n][k] * X[n] for n in range(N)]
                stn = [sum(col) for col in zip(*cols)]
                sbn = sum([self.resps[n][k] for n in range(N)])

                self.centers[k] = stn / sbn

            # Calculate costs
            cost = self.calc_cost(X, N)

            if len(self.costs) > 0 and self.costs[-1] - cost < 0.0001:
                print('Converged at iteration', len(self.costs))
                print('Final cost:', cost)
                plt.plot(self.costs)
                plt.show()
                break

            self.costs.append(cost)

        return self.centers

    def calc_responsibilities(self, X):

        N = X.shape[0]

        self.resps = np.zeros((N, self.K))

        for n in range(N):
            x = X[n]

            sk = sum([np.exp(-self.beta * self.distance(self.centers[k], x))
                      for k in range(self.K)])

            for k in range(self.K):
                c = self.centers[k]
                d = self.distance(c, x)

                self.resps[n][k] = np.exp(-self.beta * d) / sk

    def distance(self, a, b):
        x1, y1 = a
        x2, y2 = b

        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def calc_cost(self, X, N):
        return sum([self.resps[n][k] * self.distance(self.centers[k], X[n])
                    for n in range(N)
                    for k in range(self.K)])

    def predict(self, X):
        self.calc_responsibilities(X)
        return np.argmax(self.resps, axis=1)


def get_data():
    D = 2 # Two-dimensional data to easily visualize
    s = 4 # Separation between clusters
    m1 = np.array([0, 0])
    m2 = np.array([s, s])
    m3 = np.array([0, s])

    N = 600
    X = np.zeros((N, D))
    X[:200, :] = 0.5*np.random.randn(200, D) + m1
    X[200:400, :] = 0.5*np.random.randn(200, D) + m2
    X[400:, :] = 0.5*np.random.randn(200, D) + m3

    return X


def main():
    X = get_data()

    # Plot the original data
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    SKM = SoftKMeansClustering(K=3, beta=1)

    centers = SKM.fit(X)
    labels = SKM.predict(X)

    # Plot the data with colors representing cluster assignments
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.show()


if __name__ == '__main__':

    main()
