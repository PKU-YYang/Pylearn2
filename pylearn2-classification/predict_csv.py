#!/usr/bin/env python
# coding: utf-8
"""
Script to predict values using a pkl model file.

This is a configurable script to make predictions.

Basic usage:

.. code-block:: none

    predict_csv.py pkl_file.pkl test.csv output.csv

Optionally it is possible to specify if the prediction is regression or
classification (default is classification). The predicted variables are
integer by default.
Based on this script: http://fastml.com/how-to-get-predictions-from-pylearn2/.
This script doesn't use batches. If you run out of memory it could be 
resolved by implementing a batch version.

"""
from __future__ import print_function

__authors__ = ["Zygmunt Zając", "Marco De Nadai"]
__license__ = "GPL"



#用法：
#python predict_csv.py softmax_regression_best.pkl adult/test.csv output.txt -H
#test.csv因为第一列有表头，所以要-H
#这个函数会同时输出prob_matrix和label


import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch a prediction from a pkl file"
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('test_filename',
                        help='Specifies the csv file with the values to predict')
    parser.add_argument('output_filename',
                        help='Specifies the predictions output file')
    parser.add_argument('--prediction_type', '-P',
                        default="classification",
                        help='Prediction type (classification/regression)')
    parser.add_argument('--output_type', '-T',
                        default="int",
                        help='Output variable type (int/float)')
    parser.add_argument('--has-headers', '-H',
                        dest='has_headers',
                        action='store_true',
                        help='Indicates the first row in the input file is feature labels')
    parser.add_argument('--has-row-label', '-L',
                        dest='has_row_label',
                        action='store_true',
                        help='Indicates the last column in the input file is row labels')
    return parser

def predict(model_path, test_path, output_path, predictionType="regression", outputType="float",
            headers=True, last_col_label=False):
    """
    Predict from a pkl file.

    Parameters
    ----------
    modelFilename : str
        The file name of the model file.
    testFilename : str
        The file name of the file to test/predict.
    outputFilename : str
        The file name of the output file.
    predictionType : str, optional
        Type of prediction (classification/regression).
    outputType : str, optional
        Type of predicted variable (int/float).
    headers : bool, optional
        Indicates whether the first row in the input file is feature labels
    first_col_label : bool, optional
        Indicates whether the first column in the input file is row labels (e.g. row numbers)
    """

    print(predictionType,outputType)

    print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    print("setting up symbolic expressions...")

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X) #label
    M = model.fprop(X) #Prob

    if predictionType == "classification":
        Y = T.argmax(Y, axis=1) #取每行的最大值

    f = function([X], Y,allow_input_downcast=True)

    f_prob = function([X], M,allow_input_downcast=True)

    print("loading data and predicting...")

    # x is a numpy array
    # x = pickle.load(open(test_path, 'rb'))


    if headers:
        skiprows=1
    else:
        skiprows=0

    #print(headers,'skiprow=',skiprows)
    x = np.loadtxt(test_path, delimiter=',', skiprows=skiprows)

    if last_col_label:
        x = x[:,:-1]

    y = f(x)

    if predictionType == "classification":

        m=f_prob(x)

    print("writing predictions...")

    variableType = "%d"
    if outputType != "int":
        variableType = "%f"

    #print(variableType)
    np.savetxt(output_path, y, fmt=variableType)
    np.savetxt("".join(["Prob_",output_path]), m, fmt="%f")
    return True

if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename, args.test_filename, args.output_filename,
        args.prediction_type, args.output_type,
        args.has_headers, args.has_row_label)
    if not ret:
        sys.exit(-1)

