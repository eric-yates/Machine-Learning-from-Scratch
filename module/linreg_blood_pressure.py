"""
This module uses linear regression to make predictions for the blood
pressure of 11 patients. The data is accessed from an Excel file called
'mlr02.xls' that has the following format:
    
    X1 = systolic blood pressure
    X2 = age in years
    X3 = weight in pounds

This module uses a LinearRegression class imported from the module
ml_models. Further documentation for this class is available within
that module.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ml_models import LinearRegression

def load_data():
    """
    Loads an Excel file into a Pandas DataFrame. Then, a bias column
    'ones' is added to the DataFrame. The targets are selected from the
    'X1' (systolic blood pressure) column. The input data are selected
    from the 'X2', 'X3', and 'ones' columns. Both the targets and input
    data are converted to Numpy arrays and returned.
    """

    df = pd.read_excel('mlr02.xls')

    df['ones'] = 1

    X = df[['X2', 'X3', 'ones']].as_matrix()
    T = df['X1'].as_matrix()

    return X, T

def visualize_data(X, T):
    """
    Used for exploratory data analysis. Creates plots of systolic blood
    pressure vs. age and systolic blood pressure vs. weight.
    """

    plt.title("Blood Pressure vs. Age")
    plt.scatter(X[:,0], T[:])
    plt.xlabel('Age (years)')
    plt.ylabel('Systolic Blood Pressure')
    plt.show()
    
    plt.title("Blood Pressure vs. Weight")
    plt.scatter(X[:,1], T[:])
    plt.xlabel('Weight (pounds)')
    plt.ylabel('Systolic Blood Pressure')
    plt.show()

    
## Start Main Body #####################################################
"""
First, load the input data X and targets T by calling load_function.

    X, T = load_data()

Then, perform some exploratory data analysis by calling visualize_data,
which creates plots of the systolic blood pressure vs. age and weight.

    visualize_data(X, T)

Next, initialize a LinearRegression object by calling the class with the
following arguments: X, T, epochs, learn_rate, reg_L1, and reg_L2.

    model = LinearRegression(X=X,
                             T=T,
                             epochs=50,
                             learn_rate=0.000001,
                             reg_L1=0,
                             reg_L2=0)

Now, create a for loop that goes through as many iterations as the
number of epochs. For each iteration within this loop, call the
make_predictions, calculate_error, and update_weights methods. This
constitutes the training of the model. The if statement prints out the
error for every 10th iteration, starting at iteration 0.

    for i in xrange(model.epochs):
        
        model.make_predictions()
        model.calculate_error()
        model.update_weights()

        if i % 10 == 0:
            print "Iteration: " + str(i) + "   Error:", model.errors[i]

Once training has finished, calculate the R-squared value to determine
the correlation.

    model.calculate_R2()

Next, plot the error by training iteration.

    model.plot_errors()

Then, plot the targets vs. predictions to visualize how good of a fit
this model is. Include xlabel and ylabel arguments to add a label to
the x-axis and y-axis, respectively.

    model.plot_predictions(x_label='Patient #', ylabel="Systolic Blood Pressure")

Finally, print out the final error and the final weights of the trained
model, with spacing in between them to increase readability.

    print ""
    print "Final Error:", model.errors[-1]
    print ""

    print "Final Weights:", model.w
    print ""
    
"""

X, T = load_data()

visualize_data(X, T)

model = LinearRegression(X=X,
                         T=T,
                         epochs=100,
                         learn_rate=0.000001,
                         reg_L1=10,
                         reg_L2=0)

for i in xrange(model.epochs):
    
    model.make_predictions()
    model.calculate_error()
    model.update_weights()

    if i % 10 == 0:
        print "Iteration: " + str(i) + "   Error:", model.errors[i]

model.calculate_R2()

model.plot_errors()
model.plot_predictions(x_label='Patient #', y_label="Systolic Blood Pressure")

print ""
print "Final Error:", model.errors[-1]
print ""

print "Final Weights:", model.w
print ""
