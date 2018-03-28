import numpy as np
import matplotlib.pyplot as plt


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
        activation_1: Inner. 'sigmoid' or 'tanh'.
        activation_2: Outer. 'softmax', 'relu', 'sigmoid',or 'tanh'.
        
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

        from ml_models import NeuralNetwork

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
                              activation_1='tanh',
                              activation_2='softmax')

        for i in xrange(model.epochs):

            model.i = i

            model.forwardprop()
            
            model.check_accuracy()
            model.cross_entropy()
            
            model.backprop() 

            if i % 10 == 0:

                print i
                print "Train Accuracy %: ", model.train_accuracies[i]
                print "Test Accuracy %:  ", model.test_accuracies[i]

                print "Train Cost: ", model.train_costs[i]
                print "Test Cost:  ", model.test_costs[i]
                print ""   
    """

    def __init__(self, X=None, T=None, L=1, M=10, proportion_train=0.8,
                 epochs=1000, learn_rate=0.0001, reg_L1=0, reg_L2=0,
                 activation_1='sigmoid', activation_2='softmax'):
        """
        Sets the input data X and targets T. Sets the number of samples
        N, the number of features D, and the number of classes K. Sets
        the neural network architecture by number of hidden layers L and
        number of hidden units per layer M.

        Initalizes empty lists to store train/test errors and train/test
        accuracies. Initializes other empty lists to store the values of
        weights/bias terms at the maximum train/test accuracies.

        Sets the hyperparameters and chooses an activation for both the
        inner and outer functions. In case of one hidden layer:

            Y=Outer( Inner(XW[0] + B[) * V + c).

        Calls the prepare_data method to create train/test data,
        indicator matrices, and target labels. Calls the init_weights
        method to initialize the weights.
        """

        self.X = X
        self.T = T

        self.Ntotal, self.D = self.X.shape

        self.K = int(max(self.T)) + 1

        self.L = L
        self.M = M

        self.train_costs = []
        self.test_costs = []

        self.train_accuracies = []
        self.test_accuracies = []

        self.max_train = [0, 0, None, None]
        self.max_test = [0, 0, None, None]

        self.i = 0
        self.proportion_train = proportion_train
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.reg_L1 = reg_L1
        self.reg_L2 = reg_L2
        
        self.activation_1 = activation_1
        self.activation_2 = activation_2

        self.prepare_data()
        self.init_matrices()

    def prepare_data(self):
        """
        Splits the inputs X and targets T into train and test sets based
        on the proportion rate, creates indicator matrices of targets
        from the lists of targets, and creates target labels containing
        the index in K of the target. The target labels are used to check
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
        into a matrix of targets T_ind [N x K]. Used to make predictions
        from categorical variables.
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
        'input predictions,' or more simply just the input matrices X,
        for train and test sets respectively. This was done to simplify
        the recursive calculation of cost function differentials in the
        method calc_diffs.
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

            if self.activation_1 == 'sigmoid':
                self.Ztrain[i+1] = self.sigmoid(self.Ztrain[i].dot(self.W[i]) + self.B[i])
                self.Ztest[i+1] = self.sigmoid(self.Ztest[i].dot(self.W[i]) + self.B[i])

            elif self.activation_1 == 'tanh':
               self.Ztrain[i+1] = self.tanh(self.Ztrain[i].dot(self.W[i]) + self.B[i])
               self.Ztest[i+1] = self.tanh(self.Ztest[i].dot(self.W[i]) + self.B[i])

            else:
                print "Error: Invalid activation_1 function."
                print "Please use 'sigmoid' or 'tanh'."
                
        if self.activation_2 == 'softmax':
            self.Ytrain = self.softmax(self.Ztrain[self.L].dot(self.W[self.L]) + self.B[self.L])
            self.Ytest = self.softmax(self.Ztest[self.L].dot(self.W[self.L]) + self.B[self.L])

        elif self.activation_2 == 'sigmoid':
            self.Ytrain = self.sigmoid(self.Ztrain[self.L].dot(self.W[self.L]) + self.B[self.L])
            self.Ytest = self.sigmoid(self.Ztest[self.L].dot(self.W[self.L]) + self.B[self.L])

        elif self.activation_2 == 'tanh':
            self.Ytrain = self.tanh(self.Ztrain[self.L].dot(self.W[self.L]) + self.B[self.L])
            self.Ytest = self.tanh(self.Ztest[self.L].dot(self.W[self.L]) + self.B[self.L])

        elif self.activation_2 == 'relu':
            self.Ytrain = self.relu(self.Ztrain[self.L].dot(self.W[self.L]) + self.B[self.L])
            self.Ytest = self.relu(self.Ztest[self.L].dot(self.W[self.L]) + self.B[self.L])

        else:
            print "Error: Invalid activation_2 function."
            print "Please use 'softmax', 'sigmoid', 'relu', or 'tanh'."


    def softmax(self, A):
        """
        Returns the softmax of the output of the final hidden layer to
        create predictions. Because the sum of the values for all
        classes is exactly 1, the values can be interpreted as
        probability values that a sample belongs in a given class. Thus,
        it is well-suited for multi-class classification.
        """

        e_a = np.exp(A)

        return e_a / e_a.sum(axis=1, keepdims=True)

    def sigmoid(self, A):
        """
        Returns the sigmoid, which ranges between 0 and 1 and is
        centered around y=0.5. Well-suited for when the desired output
        is a probability value.
        """
        return 1 / (1 + np.exp(-A))

    def relu(self, A):
        """
        Returns the rectified linear unit (relu), which is:
        
            x for x>0
            0 for x<=0

        The relu function is very computationally efficient. However,
        it may lead to many "dead neurons" if the learning rate is set
        too high. Once the gradient reaches zero, it stays there. So,
        the neuron still is executed but makes no difference on
        predictions. Setting a lower learning rate or using other
        approaches such as 'leaky relu' may solve this.
        """

        return A * (A > 0)
        
    def tanh(self, A):
        """
        Returns the hyperbolic tangent, which ranges between -1 and 1
        and is centered around y=0. The tanh is a scaled and shifted
        form of a sigmoid function.
        """

        return 2.0 / (1 + np.exp(-2*A)) - 1.0

    def backprop(self):
        """
        First, calculates the differentials of each neuron wrt the error
        between predictions and targets. Then, updates the values of
        each weight and each bias term of each neuron.

        The model takes a gradient descent approach to optimizing the
        weights. The derivative of the cost function with respect to the
        weights dJ/dw for weight matrix i (after calling calc_diffs) is:

            [1]:   self.Jw[i]
        
        L1 (Lasso) regularization encourages sparsity of weights and
        helps to prevent overfitting. This is done by subtracting the
        sign of weights times the L1 regularization constant from the
        derivative of the cost function:

            [2]:   [1] - self.reg_L1 * np.sign(self.W[i])

        L2 (Ridge) regularization encourages small weights for all
        weights and also helps to prevent overfitting. This is done by
        subtracting the L2 regularization times two times the weights
        from the derivative of the cost function.

            [3]:   [2] - self.reg_L2 * 2*self.W[i]

        Finally, the weights are updated by adding to each weight the
        learning rate times the derivative of the cost function minus
        the regularization terms:

            [4]:   self.W[i] += self.learning_rate * [3]

        A similar process is done to optimize the bias terms.
        """

        self.calc_diffs(l=0)
        
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
        the weights dJ/dW[i]. The differentials are used in gradient
        descent for an arbitratily sized neural network architecture.
        The following equation (approximately) describes this process:

        dJ/dW[L] = Jw[L] = (Y - T) Z[L]

        dJ/dW[L-1] = (Y - T) Z[L] W[L(1 - Z[L])Z[L-1],...,W[i+1] Z[i+1](1- Z[i+1])Z[i]

        The equation above misses slight nuances, which are all taken
        into account in the actual code below. Once the method is
        called, it starts with l=0. It then calls itself with l=1, then
        l=2,..., until l=N.

        At that point, it calculates the differential of the final
        hidden layer directly from the predictions and targets and is
        saved to Jw[L] and Jb[L]. This result is then passed back to
        the previous hidden layers and used to calculate their
        differentials until the first hidden layer is reached.
        """
        
        if l == self.L:

            dZ = self.Ytrain - self.Ttrain_ind
            self.Jw[self.L] = self.Ztrain[self.L].T.dot(dZ)
            self.Jb[self.L] = dZ.sum(axis=0)

            return dZ

        else:
            
            dZ = self.calc_diffs(l+1).dot(self.W[l+1].T) * (1 - self.Ztrain[l+1] * self.Ztrain[l+1])
            self.Jw[l] = self.Ztrain[l].T.dot(dZ)
            self.Jb[l] = dZ.sum(axis=0)

            return dZ

    def cross_entropy(self):
        """
        Calculates the costs using a cross-entropy approach.

            J = -T logY / N
        """

        cost_train = -(self.Ttrain_ind * np.log(self.Ytrain)).sum() / self.Ntrain
        
        self.train_costs.append(cost_train)
        
        cost_test = -(self.Ttest_ind * np.log(self.Ytest)).sum() / self.Ntest
        
        self.test_costs.append(cost_test)

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
        """

        if current_acc > max_acc[1]:
            return [self.i, current_acc, self.W, self.B]

        else:
            return max_acc

    def plot_costs(self):
        """
        Plots the train error and test error against iterations. Can be
        used to visualize when to stop training to prevent overfitting
        once test error begins to rise.
        """

        plt.title("Train vs. Test Error")
        plt.plot(self.train_costs, label="Train")
        plt.plot(self.test_costs, label="Test")
        plt.legend()
        plt.show()

    def plot_accuracies(self):
        """
        Plots the train and test accuracies against iterations. This is
        a way to visualize the degree of overfitting pre
        """

        plt.title("Train vs. Test Accuracies")
        plt.plot(self.train_accuracies, label="Train")
        plt.plot(self.test_accuracies, label="Test")
        plt.legend()
        plt.show()
