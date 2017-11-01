"""
Author: Eric Yates

This module contains templates for various machine learning models. All
models are built from scratch using the Numpy library. Matplotlib.pyplot
is used to visualize data and results. The models have not been
optimized and are meant for educational purposes rather than optimal
efficiency.

I originally started learning about these models from courses on Udemy
by the instructor 'LazyProgrammer.' He explained the theory and
mathematics behind the models first, then showed the implementation in
code. The courses are an excellent introduction to understand the
fundamental principles and mathematics behind machine learning.

In all cases, the models here have been expanded upon from the example
code LazyProgrammer provided. For example, I made all models
object-oriented and added in-depth documentation. In the NeuralNetwork
class, I added functionality to choose the number of hidden layers,
whereas LazyProgrammer showed an example with one fixed hidden layer.

To use this module, save this file in the same directory as the project
file you are working with. Then, add the following line:

    from ml_models import LinearRegression, LogisticRegression,...

where import contains the models you would like to import. Currently
supported models are:

    LinearRegression: Used for numerical regression.
    LogisticRegression: Used for binary classification.
    NeuralNetwork: Used for multi-class classification.

For all models, it is assumed that the model receives well-prepared and
cleaned input data X and targets T. Any feature engineering should be
done prior to creating a model.
"""

import numpy as np
import matplotlib.pyplot as plt


########################################################################
## Start of LinearRegression ###########################################


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
        X [N x D]: All input samples and features.      
        T [N x 1]: All output targets.
        Y [N x 1]: All output predictions.     
        w [D x 1]: Weights for each input feature.

    To implement the LinearRegression class, insert code that looks
    something like the following:

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

            if i % 10 == 0:
                print "Iteration: " + str(i) + "   Error: ", model.errors[i]

            model.update_weights()

        model.caclulate_R2()
        
        model.plot_errors()
        model.plot_predictions(ylabel="[Quantity Being Predicted]")
    """

    def __init__(self, X=None, T=None, epochs=100, learn_rate=0.001,
                 reg_L1=0, reg_L2=0):
        """
        Sets the input data and targets, sets the number of samples N
        and the number of features D, initalizes an empty list to store
        errors, sets the hyperparameters, and calls the init_weights
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
        Initializes the weights by picking randomly from a normal
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
        error = format(error, '.2f')
        
        self.errors.append(error)

    def update_weights(self):
        """
        The derivative of the cost function with respect to the weights
        is the transpose of X times the difference between targets T and
        predictions Y:

            [1]:   self.X.T.dot(T - self.Y)
        
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
        Calculates the R-squared correlation between targets and
        predictions to measure the fit of the model.
        """

        d1 = self.T - self.Y
        d2 = self.T - self.T.mean()

        self.r2 = 1 - d1.dot(d1) / d2.dot(d2)
        self.r2 = format(self.r2, '.3f')

        print "R2:", self.r2

    def plot_errors(self):
        """
        Plots the MSE (mean squared errors) against iteration i.
        """

        plt.title("Prediction Error")
        plt.plot(self.errors)
        plt.ylabel("MSE (Mean Squared Error)")
        plt.xlabel("Iteration")
        plt.show()

    def plot_predictions(self, y_label=None):
        """
        Plots the targets and predictions to visualize how well of a fit
        the model is.
        """

        plt.title("Targets vs. Predictions")
        plt.plot(self.T, label="Targets")
        plt.plot(self.Y, label="Predictions")
        plt.ylabel(y_label)
        plt.xlabel("Sample")
        plt.legend()
        plt.show()


## End of LinearRegression #############################################
## Start of LogisticRegression #########################################


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

            model.cross_entropy()

            if i % 100 == 0:
                print "Iteration: " + str(i) + " Error: ", model.current_error

            model.update_weights()
    """

    def __init__(self, X=None, T=None, epochs=1000,
                 learn_rate=0.001, reg_L1=0, reg_L2=0):
        """
        Sets the input data and targets, sets the hyperparameters, and
        calls the init_weights method.
        """
        
        self.X = X
        self.T = T
        self.N, self.D = self.X.shape
                
        self.errors = []
        self.accuracies = []
        
        self.epochs = epochs
        self.learning_rate = learning_rate
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
        The derivative of the cost function with respect to the weights
        is the transpose of X times the difference between targets T and
        predictions Y:

            [1]:   self.X.T.dot(T - self.Y)
        
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
        The output of each logistic unit is equal to the multiplication
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
        that the sigmoid values for all sample varies between 0 and 1.
        The value of the sigmoid for each sample then represents the
        probability that the sample fits into a given class. 
        """
        
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self):
        """
        The cost function is assumed to be a cross-entropy function.
        This method calculates the cross-entropy cost at each given
        iteration. These costs are used to create a graph that shows
        the error vs. training iteration.
        """
        
        error = 0
        
        for i in xrange(self.N):
            if self.T[i] == 1:
                error -= np.log(self.Y[i])
            else:
                error -= np.log(1 - self.Y[i])
                
        self.errors.append(error)

    def check_accuracy(self):
        """
        If a predicted value in Y is greater than or equal to 0.5, it is
        predicted to belong to class '1', and if the predicated value in
        Y is less than 0.5, it is predicted to belong to class '0'. The
        accuracy then reflects the percentage of predictions that
        correctly match targets.
        """
        
        accuracy = 100*(1 - np.abs(self.T - np.round(self.Y)).sum()/ self.N)

        self.accuracies.append(accuracy)

    def plot_error(self):
        """
        Plots the total error against iteration i.
        """

        plt.title("Error")
        plt.plot(self.errors)
        plt.show()


## End of LogisticRegression ###########################################
## Start of NeuralNetwork ##############################################


class NeuralNetwork:
    """
    Creates a neural network with L hidden layers and M hidden units.
    
    Architecture Constants:
        N: Number of input samples.
        D: Number of input features.
        K: Number of output classes.

    Architecture Variables:
        L: Number of hidden layers. Must be >= 1.
        M: Number of hidden units per hidden layer. Must be >= 1.

    Hyperparameters:
        proportion_train: Proportion of training data to test data.
        epochs: Number of training iterations.
        learn_rate: Rate of change used in gradient descent.
        reg_L1: Sets the penalty for L1 (Lasso) regularization.
        reg_L2: Sets the penalty for L2 (Ridge) regularization.
        activation: Choose from 'softmax', 'sigmoid', 'relu', 'tanh'.
        
    Numpy Matrices:
        X [N x D]: All input samples and features.      
        T [N x 1]: All output targets.
        T_ind [N x K]: An indicator matrix of the targets.
        T_labels [N x 1]: A list containing the index of each target.
        Z [Varies]: All values for each hidden unit of each hidden layer.
        Y [N x K]: All output predictions.
        
        W [Varies]: Weights for each hidden unit of each hidden layer.
        B [Varies]: Bias terms for each hidden unit of each hidden layer.
        Jw [Varies]: Weight differential wrt cost.
        Jb [Varies]: Bias term differential wrt cost.

    To implement the NeuralNetwork class, insert code that looks
    something like the following:

        X = ...
        T = ...

        model = NeuralNetwork(X=X,
                              T=T,
                              L=1,
                              M=20,
                              proportion_train=0.8,
                              epochs=1000,
                              learn_rate=10e-6,
                              reg_L1=0,
                              reg_L2=0,
                              activation='softmax')

        for i in xrange(model.epochs):

            model.i = i

            model.forwardprop()
            model.cross_entropy()
            model.check_accuracy()

            if i % 10 == 0:

                print i
                print "Train Accuracy %: ", model.train_accuracies[i]
                print "Test Accuracy %:  ", model.test_accuracies[i]

                print "Train Cost: ", model.train_costs[i]
                print "Test Cost:  ", model.test_costs[i]
                print ""

            model.backprop()    
    """

    def __init__(self, X, T, L, M, proportion_train, epochs,
                 learn_rate, reg_L1, reg_L2, activation):
        """
        Sets the input data and targets, specifies the neural network
        architecture, sets the hyperparameters, calls the prepare_data
        method, and calls the init_matrices method.
        """

        self.X = X
        self.T = T

        self.Ntotal, self.D = self.X.shape

        self.K = int(max(self.T)) + 1

        self.L = L
        self.M = M

        self.train_costs = []
        self.test_costs = []

        self.max_train = [0, 0, None, None]
        self.max_test = [0, 0, None, None]

        self.train_accuracies = []
        self.test_accuracies = []

        self.i = 0
        self.proportion_train = proportion_train
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.reg_L1 = reg_L1
        self.reg_L2 = reg_L2
        self.activation = activation

        self.prepare_data()
        self.init_matrices()

    def prepare_data(self):
        """
        Splits the inputs & targets data into train & test sets based on
        the proportion rate, creates indicator matrices of targets from
        the lists of targets, and creates target labels containing the
        index in K of the target. The target labels are used to check
        the accuracy of predictions.
        """
        
        self.Ntrain = int(self.proportion_train * self.Ntotal)
        self.Ntest = self.Ntotal - self.Ntrain

        self.Xtrain = self.X[:self.Ntrain]
        self.Ttrain = self.T[:self.Ntrain]
        self.Xtest = self.X[self.Ntrain:]
        self.Ttest = self.T[self.Ntrain:]

        self.Ttrain_ind = self.targets_to_indicator(self.Ttrain)
        self.Ttest_ind = self.targets_to_indicator(self.Ttest)

        self.Ttrain_labels = np.argmax(self.Ttrain_ind, axis=1)
        self.Ttest_labels = np.argmax(self.Ttest_ind, axis=1)

    def targets_to_indicator(self, T):
        """
        Uses one-hot encoding to transform a list of targets T [N x 1]
        into a matrix of targets T_ind [N x K].
        """

        n = len(T)

        T_ind = np.zeros((n, self.K))

        for j in xrange(n):
            T_ind[j][int(T[j])] = 1
                
        return T_ind

    def init_matrices(self):
        """
        The weights and bias matrices are intialized by randomly
        sampling from a Normal distribution with mean = 0 and
        std = 1/sqrt(# features). This prevents initial saturation
        of neurons.

        The other matrices are initialized to zeros matrices because
        they are not used before being filled in by other methods.

        The data structure of each matrix is technically a list,
        where the index of the list correponds to the nth element.

        Example self.W = [W0, W1, W2,...WL]
            W0 is the weight matrix from input to first hidden layer.
            W1 is the weight matrix from first hidden layer to second.
            W2 is the weight matrix from second hiden layer to third.
            ...
            WL is the weight matrix from final hidden layer to output.

            where L is the total number of hidden layers.

        The 0th element of both Ztrain and Ztest can be thought of as
        'input predictions,' or more simply just the input matrices
        themselves, for train and test sets respectively. This was done
        to simplify the recursive calculation of cost function
        differentials in the method calc_diffs.
        """
        
        self.W = [np.random.randn(self.D, self.M) / np.sqrt(self.D + self.M)] \
                + [np.random.randn(self.M, self.M) / np.sqrt(2*self.M)] * (self.L-1) \
                + [np.random.randn(self.M, self.K) / np.sqrt(self.M + self.K)]

        self.B = [np.random.randn(self.M) / np.sqrt(self.M)] * self.L \
                + [np.random.randn(self.K) /  np.sqrt(self.K)]
        
        self.Ztrain = [np.zeros((self.Ntrain, self.D))] \
                    + [np.zeros((self.Ntrain, self.M))] * self.L
        
        self.Ztest = [np.zeros((self.Ntest, self.D))] \
                    + [np.zeros((self.Ntest, self.M))] * self.L

        self.Jw = [np.zeros((self.D, self.M))] \
                + [np.zeros((self.M, self.M))] * (self.L - 1) \
                + [np.zeros((self.M, self.K))]

        self.Jb = [np.zeros(self.M)] * self.L \
                + [np.zeros(self.K)]

        self.Ztrain[0] = self.Xtrain

        self.Ztest[0] = self.Xtest

        self.Ytrain = [np.zeros((self.Ntrain, self.K))]

        self.Ytest = [np.zeros((self.Ntest, self.K))]

    def forwardprop(self):
        """
        Calculates the output values for each neuron, provides the
        previous output value for each neuron as the input to the next
        hidden layer, and finally puts the output values of the final
        hidden layer through a softmax function to create predictions.
        """

        for i in xrange(self.L):
            self.Ztrain[i+1] = np.tanh(self.Ztrain[i].dot(self.W[i]) + self.B[i])
            self.Ztest[i+1] = np.tanh(self.Ztest[i].dot(self.W[i]) + self.B[i])

        Atrain = self.Ztrain[self.L].dot(self.W[self.L]) + self.B[self.L]
        Atest = self.Ztest[self.L].dot(self.W[self.L]) + self.B[self.L]

        if self.activation == 'softmax':
            self.Ytrain = self.softmax(Atrain)
            self.Ytest = self.softmax(Atest)

        elif self.activation == 'sigmoid':
            self.Ytrain = self.sigmoid(Atrain)
            self.Ytest = self.sigmoid(Atest)

        elif self.activation == 'relu':
            self.Ytrain = self.relu(Atrain)
            self.Ytest = self.relu(Atest)

        elif self.activation == 'tanh':
            self.Ytrain = self.tanh(Atrain)
            self.Ytest = self.tanh(Atest)

        else:
            print "Error: Invalid activation function."
            print "Please use 'softmax', 'sigmoid', 'relu', or 'tanh'."


    def softmax(self, A):
        """
        Returns the softmax of the output of the final hidden layer to
        create predictions.
        """

        e_a = np.exp(A)

        return e_a / e_a.sum(axis=1, keepdims=True)

    def sigmoid(self, A):
        """

        """

        return 1 / (1 + np.exp(-A))

    def relu(self, A):
        """

        """

        return A * (A > 0)

    def tanh(self, A):
        """

        """

        return np.sinh(A)/np.cosh(A)

    def backprop(self):
        """
        First, calculates the differentials of each neuron wrt the error
        between predictions and targets. Then, updates the values of
        each weight and each bias term of each neuron. L1 (Lasso)
        regularization may be used to encourage sparsity of weights.
        """

        self.calc_diffs(0)
        
        for i in xrange(self.L+1):
            self.W[i] -= self.learn_rate * (self.Jw[i]
                                            + self.reg_L1 * np.sign(self.W[i])
                                            + self.reg_L2 * 2*self.W[i])
                               
            self.B[i] -= self.learn_rate * (self.Jb[i]
                                            + self.reg_L1 * np.sign(self.B[i])
                                            + self.reg_L2 * 2*self.B[i])

    def calc_diffs(self, l):
        """
        Recursively calculates the differentials of cost functions wrt
        the weights dJ/dW. The differentials are used in gradient
        descent for an arbitratily sized neural network architecture.
        """
        
        if l == self.L:

            dZ = self.Ytrain - self.Ttrain_ind
            self.Jw[self.L] = self.Ztrain[self.L].T.dot(dZ)
            self.Jb[self.L] = dZ.sum(axis=0)

            return dZ

        elif l == 0:

            dZ = self.calc_diffs(l+1).dot(self.W[l+1].T) * (1 - self.Ztrain[l+1] * self.Ztrain[l+1])
            self.Jw[l] = self.Ztrain[l].T.dot(dZ)
            self.Jb[l] = dZ.sum(axis=0)

            return

        else:
            
            dZ = self.calc_diffs(l+1).dot(self.W[l+1].T) * (1 - self.Ztrain[l+1] * self.Ztrain[l+1])
            self.Jw[l] = self.Ztrain[l].T.dot(dZ)
            self.Jb[l] = dZ.sum(axis=0)

            return dZ

    def cross_entropy(self):
        """
        Calculates the costs using a cross-entropy approach, and then
        formats the values to 2 decimal places.
        """

        cost_train = -(self.Ttrain_ind * np.log(self.Ytrain)).sum() / self.Ntrain
        cost_train = format(float(cost_train), '.2f')
        
        self.train_costs.append(cost_train)
        
        cost_test = -(self.Ttest_ind * np.log(self.Ytest)).sum() / self.Ntest
        cost_test = format(float(cost_test), '.2f')
        
        self.test_costs.append(cost_test)

    def plot_costs(self):
        """
        Plots the training vs. test costs on a graph by epoch number.

        Inputs:
        
            train_costs: A list containing the training costs. Obtained
                         by appending 'nn.cost_train' to a list for each
                         iteration. nn
        """
        
        plt.plot(self.train_costs)
        plt.plot(self.test_costs)
        plt.show()

    def check_accuracy(self):
        """
        Calculates the accuracy, ie. the percentage of correct output
        predictions that match the targets. Stop training once the test
        accuracy levels off to prevent overfitting.
        """

        train_acc = 100*sum(self.Ttrain_labels == np.argmax(self.Ytrain, axis=1))
        train_acc = format(float(train_acc) / self.Ntrain, '.2f')

        self.train_accuracies.append(train_acc)
                               
        test_acc = 100*sum(self.Ttest_labels == np.argmax(self.Ytest, axis=1))
        test_acc = format(float(test_acc) / self.Ntest, '.2f')

        self.test_accuracies.append(test_acc)

        self.max_train = self.check_max(train_acc, self.max_train)
        self.max_test = self.check_max(test_acc, self.max_test)

    def check_max(self, current_acc, max_acc):
        """
        After calculating the accuracies using the method called
        'check_accuracy', call this method on both the training accuracy
        and the test accuracy.

        This method checks to see if the current accuracy is greater
        than the maximum accuracy seen thus far. Used to tell when to
        stop training once the test accuracy levels off to prevent
        overfitting.

        Inputs:
            i: The current iteration number.
            
            current_acc: Use 'self.train_acc' & 'self.test_acc' after
                         calling 'self.check_accuracy()'.
            
            max_acc: Use a list of form [i, maximum_accuracy].
                     At iteration [i], accuracy was [maximum_accuracy].
        """

        if current_acc > max_acc[1]:
            return [self.i, current_acc, self.W, self.B]

        else:
            return max_acc

    def plot_costs(self):
        """
        Plots the train error and test error against iterations. Used to
        tell when to stop training to prevent overfitting.
        """

        plt.title("Train vs. Test Error")
        plt.plot(self.train_costs, label="Train")
        plt.plot(self.test_costs, label="Test")
        plt.legend()
        plt.show()
