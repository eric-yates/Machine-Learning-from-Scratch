import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Architecture Constants:
        N: Total number of input samples of matrix X.
        D: Number of input features of matrix X.

    Hyperparameters:
        epochs: Number of training iterations.
        learn_rate: Rate of change used in gradient descent.
        reg_L1: Sets the penalty for L1 (Lasso) regularization.
        reg_L2: Sets the penalty for L2 (Ridge) regularization.
        
    Numpy Matrices:
        X [N x D]: Input samples and features.      
        T [N x 1]: Output targets.
        Y [N x 1]: Output predictions.     
        w [D x 1]: Weights for each input feature.

    To implement the LinearRegression class, insert code that looks
    something like the following:

        from ml_models import LinearRegression

        X = ...
        T = ...

        model = LinearRegression(X=X,
                                 T=T,
                                 epochs=300,
                                 learn_rate=0.001,
                                 reg_L1=0,
                                 reg_L2=0)

        for i in xrange(model.epochs):
            
            model.make_predictions()
            model.calculate_error()
            model.update_weights()

            if i % 10 == 0:
                print "Iteration: " + str(i) + "   Error:", model.errors[i]

        model.calculate_R2()
        
        model.plot_errors()
        model.plot_predictions()
    """

    def __init__(self, X=None, T=None, epochs=1000, learn_rate=0.001,
                 reg_L1=0, reg_L2=0):
        """
        Sets the input data X and targets T, sets the number of samples
        N and the number of features D, initalizes an empty list to
        store errors, sets the hyperparameters, and calls the init_weights
        method to initialize the weights.
        """
        
        self.X = X
        self.T = T
        self.N, self.D = self.X.shape

        self.errors = []
        
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.reg_L1 = reg_L1
        self.reg_L2 = reg_L2
        
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights by randomly sampling from a normal
        distribution with mean equal to zero and standard deviation
        equal to 1 / sqrt(# of features).
        """
        
        self.w = np.random.randn(self.D) / np.sqrt(self.D)

    def make_predictions(self):
        """
        Makes predictions Y by taking the dot product of input matrix X
        and weight vector w.
        """
        
        self.Y = self.X.dot(self.w)

    def calculate_error(self):
        """
        Calculates the mean squared error between predictions Y and
        targets T according to the following equation:

            error = (Y - T)**2 / N

        Then, the error is formatted to two significant figures and
        appended to a list called 'self.errors' to store the errors
        over iterations for a plot.
        """
        
        delta = self.Y - self.T
        error = delta.dot(delta) / self.N
        error = format(error, '.5f')
        
        self.errors.append(error)

    def update_weights(self):
        """
        The model takes a gradient descent approach to optimizing the
        weights. The derivative of the cost function with respect to the
        weights dJ/dw is the transpose of X times the difference between
        targets T and predictions Y:

            [1]:   self.X.T.dot(self.T - self.Y)
        
        L1 (Lasso) regularization encourages sparsity of weights and
        helps to prevent overfitting. This is done by subtracting the
        sign of weights times the L1 regularization constant from the
        derivative of the cost function:

            [2]:   [1] - self.reg_L1 * np.sign(self.w)

        L2 (Ridge) regularization encourages small weights for all
        weights and also helps to prevent overfitting. This is done by
        subtracting the L2 regularization times two times the weights
        from the derivative of the cost function.

            [3]:   [2] - self.reg_L2 * 2*self.w

        Finally, the weights are updated by adding to each weight the
        learning rate times the derivative of the cost function minus
        the regularization terms:

            [4]:   self.w += self.learning_rate * [3]
        """


        self.w += self.learn_rate * (self.X.T.dot(self.T - self.Y)
                                     - self.reg_L1 * np.sign(self.w)
                                     - self.reg_L2 * 2*self.w)

    def calculate_R2(self):
        """
        Calculates the R-squared correlation between targets T and
        predictions Y to measure the fit of the model.
        """

        d1 = self.T - self.Y
        d2 = self.T - self.T.mean()

        self.r2 = 1 - d1.dot(d1) / d2.dot(d2)
        self.r2 = format(self.r2, '.3f')

        print ""
        print "R2:", self.r2

    def plot_errors(self):
        """
        Plots the MSE (mean squared error) against iteration i.
        """

        plt.title("Prediction Error")
        plt.plot(self.errors)
        plt.ylabel("MSE (Mean Squared Error)")
        plt.xlabel("Iteration")
        plt.show()

    def plot_predictions(self):
        """
        Plots the targets T and predictions Y to visualize how good of
        a fit the model is.
        """

        plt.title("Targets vs. Predictions")
        plt.plot(self.T, label="Targets")
        plt.plot(self.Y, label="Predictions")
        plt.xlabel("Sample number")
        plt.legend()
        plt.show()
