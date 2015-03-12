#-*-:encoding:utf-8 -*-
__author__ = 'Yang'
# We'll need the csv module to read the file
import csv
# We'll need numpy to manage arrays of data
import numpy as np

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
    """
    Loads the red wine quality dataset from:

    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
    Modeling wine preferences by data mining from physicochemical properties.
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

    The dataset contains 1,599 examples, including a floating point regression
    target.

    Parameters
    ----------
    start: int
    stop: int

    Returns
    -------

    dataset : DenseDesignMatrix
        A dataset include examples start (inclusive) through stop (exclusive).
        The start and stop parameters are useful for splitting the data into
        train, validation, and test data.
    """
    with open('winequality-red.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        X = []
        y = []
        header = True
        for row in reader:
            # Skip the first row containing the string names of each attribute
            if header:
                header = False
                continue  #跳掉一次
            # Convert the row into numbers
            row = [float(elem) for elem in row]
            X.append(row[:-1])
            y.append(row[-1])  #最后一列是label/值
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1) #label必须是矩阵，不可以是vector

    X = X[start:stop, :]
    y = y[start:stop, :]

    return DenseDesignMatrix(X=X, y=y)
