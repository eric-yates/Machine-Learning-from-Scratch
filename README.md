# ml-models
A collection of various machine learning models.

Author: Eric Yates

The "ml_models.py" contains templates for various machine learning models.
All models are built from scratch using the Numpy library. Matplotlib.pyplot
is used to visualize data and results. The models have not been optimized 
and are meant for educational purposes rather than maximum performance.

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
