"""
This module solves the classic XOR problem. The "exclusive OR" gate
returns True if and only if one of the inputs is True. That is to say
for a set containing A and B, either A or B, but not A and B.

     A  B
    [0, 0]  False
    [0, 1]  True
    [1, 0]  True
    [1, 1]  False

The XOR problem is linearly inseperable, meaning a single line cannot
be drawn between inputs to classify the outputs. This can be seen by
calling the visualize_data function. To solve a linearly inseperable
problem using a logistic regression model, manual feature engineering
must be done, which is explained in more detail in the prepare_data
function.

As is seen in the 'nn_xor' module included in this repository, a
neural network is able to solve the XOR problem without the manual
feature engineering required in logistic regression. This ability to
automatically learn new features is a highly desireable feature of
neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

from ml_models import LogisticRegression

def prepare_data():
    """
    A matrix X initalizes the 4 cases of an XOR gate. The targets are
    equal to 1 if the result of XOR is True (either A or B, but not A
    and B) and 0 if the result is False (either not A or B or A and B).

    A bias column 'ones' and a feature engineered column 'xy' is the
    multiplication of the multiplication of the inputs. Without the
    manual feature engineering, the logistic model is not able to solve
    the problem, that is it will achieve 50% accuracy - the same as
    guessing. With the xy feature added, the model achieves 100%
    accuracy.
    """

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])

    T = np.array([0, 1, 1, 0])

    ones = np.array([[1] * 4]).T

    xy = np.matrix(X[:, 0] * X[:, 1]).T

    Xb = np.array(np.concatenate((X, xy, ones), axis=1))

    return Xb, T

def visualize_data(X):
    """
    Plots the inputs to a logic gate and colors them by the output class
    (True or False). Notice that no single line can be drawn between the
    two different classes, meaning it is a linearly inseperable problem.
    """

    plt.title("XOR Problem")
    plt.scatter(X[1:3, 0], X[1:3, 1], c='green', label='True')
    plt.scatter(X[(0,3), 0], X[(0,3), 1], c='blue', label='False')
    plt.xlabel("Gate A")
    plt.ylabel("Gate B")
    plt.axis('equal')
    plt.legend()
    plt.show()
    
def main():
    """
    First, create the input data X and targets T. Then, visualize the
    XOR problem to see how it is a linearly inseperable problem. Next,
    initialize an object of class LogisticRegression.

    Enter a for loop with as many iterations as the epochs
    hyperparameter for the training of the model. Within the loop,
    make predictions, calculate the cross-entropy & accuracy, and
    update the weights by comparing predictions to targets. If the
    iteration is a multiple of 100, print out the current error and
    the current accuracy.

    Once training is over, print the final weights and the final
    accuracy. Finally, plot the errors vs. iteration to see the how
    quickly the model learns.
    """
    
    X, T = prepare_data()

    visualize_data(X)

    model = LogisticRegression(X=X,
                               T=T,
                               epochs=3000,
                               learn_rate=0.01,
                               reg_L1=0.001,
                               reg_L2=0)

    for i in xrange(model.epochs):

        model.make_predictions()
        model.cross_entropy()
        model.check_accuracy()
        model.update_weights()

        if i % 100 == 0:
            print "Iteration: " + str(i)
            print "Error: ", model.errors[i]
            print "Accuracy: " + str(model.accuracies[i]) + "%\n"

    print "Final Weights: ", model.w
    print "Final Accuracy: " + str(model.accuracies[i]) + "%"

    model.plot_error()

if __name__ == "__main__":
    main()
