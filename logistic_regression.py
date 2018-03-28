import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    This logistic regression model assumes a binary classification, such
    as yes/no, included/excluded,...
    
    Architecture Constants:
        N: Number of input samples.
        D: Number of input features.

    Hyperparameters:
        epochs: Number of training iterations.
        learn_rate: Rate of change used in gradient descent.
        reg_L1: Sets the penalty for L1 (Lasso) regularization.
        reg_L2: Sets the penalty for L2 (Ridge) regularization.
        
    Numpy Matrices:
        X [N x D]: All input samples and features.      
        T [N x 1]: All output targets.
        Y [N x 1]: All output predictions.     
        w [D x 1]: Weights for each input feature.

    To implement the LogisticRegression class, insert code that looks
    something like the following:

        from ml_models import LogisticRegression

        X = ...
        T = ...

        model = LogisticRegression(X=X,
                                   T=T,
                                   epochs=1000,
                                   learn_rate=0.001,
                                   reg_L1=0,
                                   reg_L2=0)

        for i in xrange(model.epochs):

            model.make_predictions()
            
            model.check_accuracy()
            model.cross_entropy()
            
            model.update_weights()

            if i % 10 == 0:
                print "Iteration: " + str(i) + " Error: ", model.errors[i]
                print "Accuracy:", model.accuracies[i]

        model.plot_errors()
    """

    def __init__(self, X=None, T=None, epochs=1000, learn_rate=0.001,
                 reg_L1=0, reg_L2=0):
        """
        Sets the input data X and targets T, sets the number of samples
        N and the number of features D, initalizes an empty list to
        store errors, initializes an empty list to store accuracies,
        sets the hyperparameters, and calls the init_weights method to
        initialize the weights.
        """
        
        self.X = X
        self.T = T
        self.N, self.D = self.X.shape
                
        self.errors = []
        self.accuracies = []
        
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.reg_L1 = reg_L1
        self.reg_L2 = reg_L2

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights by picking randomly from a normal
        distribution with mean equal to zero and standard deviation
        equal to 1 / sqrt(# of features).
        """
        
        self.w = np.random.randn(self.D) / np.sqrt(self.D)

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
        

    def make_predictions(self):
        """
        The output of each logistic unit is equal to the dot product
        of input data X with the weight vector w:

            [1]:   self.X.dot(self.w)

        Then, the output of each logistic unit is passed through a
        sigmoid function to make predictions Y:

            [2]:   self.Y = self.sigmoid([1])
        """
        
        self.Y = self.sigmoid(self.X.dot(self.w))

    def sigmoid(self, z):
        """
        The sigmoid function takes input from the output of each
        logistic unit and flattens the value between 0 and 1, such
        that the sigmoid values for all samples varies between 0 and 1.
        The value of the sigmoid for each sample then represents the
        probability that the sample fits into a given class. 
        """
        
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self):
        """
        The cost function J is assumed to be a cross-entropy function.
        
            J = t logy + (1-t)log(1-y)

        Since the target is either 0 or 1 (yes/no, include/exclude,...),
        only one of these terms contributes to the error for each given
        sample. If the target is 1, the contribution to error is:

            error -= np.log(self.Y[i])

        If the target is 0, the contribution to error is:

            error -= np.log(1 - self.Y[i])
            
        Iterate through all samples N by comparing predictions Y to
        targets T. Once all iterations are complete, append the total
        error divided by N to the list 'errors' to see error as the
        model trains.
        """
        
        error = 0
        
        for i in xrange(self.N):
            if self.T[i] == 1:
                error -= np.log(self.Y[i])
            else:
                error -= np.log(1 - self.Y[i])
                
        self.errors.append(error/self.N)

    def check_accuracy(self):
        """
        If a predicted value in Y is greater than or equal to 0.5, it is
        predicted to belong to class '1', and if the predicated value in
        Y is less than 0.5, it is predicted to belong to class '0'. The
        accuracy then reflects the percentage of predictions that
        correctly match targets. Append the accuracy for each iteration
        to a list 'accuracies' to see accuracies as the model trains.
        """
        
        accuracy = 100*(1 - np.abs(self.T - np.round(self.Y)).sum()/ self.N)

        self.accuracies.append(accuracy)

    def plot_errors(self):
        """
        Plots the total error against iteration i.
        """

        plt.title("Error")
        plt.plot(self.errors)
        plt.ylabel("Error")
        plt.xlabel("Iteration")
        plt.show()
