# ml-models

A collection of various object-oriented machine learning models built from scratch in Python.

These models have not been optimized and are meant for educational purposes rather than maximum performance.

## Getting Started

These instructions will get a copy of the project up and running on your local machine.

### Prerequisites

Instructions for installing these software are listed in the next section: Installing. These are the software packages needed to run:

* Python **2.7**

These Python packages are also needed:

* numpy
* pandas
* matplotlib


### Installing

If your computer does not already have Python **2.7** installed, download it [here](https://www.python.org/downloads/).

By default, Python should come with pip (a package manager). Use it to install the following dependencies by opening the Terminal/command line and entering the commands as follows, each line as a separate command (make sure to capitalize Tkinter):

```
pip install numpy
pip install pandas
pip install matplotlib
```

## Usage

### Basic

To use this module, save the "ml_models.py" file in the same directory as the project
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

## Built With

* [Python](https://www.python.org/about/) - A programming language used here to create exploratory data graphs
* [Numpy](http://www.numpy.org/) - Python library for mathematical and matrix operations 
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/) - Python library for data manipulation
* [Matplotlib](https://matplotlib.org/) - Python library for graphing data


## Authors

* **Eric Yates** - [Github Profile](https://github.com/eric-yates)

## License

This project is licensed under the MIT License - see the [LICENSE.md](/LICENSE.md) file for details.

## Acknowledgments

* **LazyProgrammer**: For his [courses](https://www.udemy.com/user/lazy-programmer/) on machine learning
