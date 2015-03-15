#-*-:encoding:utf-8 -*-

#注意：1 pylearn2要求Label是one_hot序列，label会被自动转换
#     2 train,test,valid三个数据里的出现的label种类必须一样多，不可以有一个文件里有个label没出现过
#     3 label可以不从0开始计数
__author__ = 'Yang'
import csv
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(path,start=0, stop=None, header=True, mode="classification"):  #有默认值的形参要从右边开始放
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
    with open(path, 'rU') as f: #如果报new-line character seen in unquoted field，就用rU
        reader = csv.reader(f, delimiter=';')
        X = []
        y = []
        #header = False
        for row in reader:
            # Skip the first row containing the string names of each attribute
            if header:
                header = False
                continue  #跳掉一次
            # Convert the row into numbers
            row = [float(elem) for elem in row]
            #print(row)
            X.append(row[:-1])
            y.append(row[-1])  #最后一列是label/值

    X = np.asarray(X)
    y = np.asarray(y)

    # read_path = np.loadtxt(path, delimiter=',', skiprows=int(header))
    # X = np.asarray(read_path[:,:-1])
    # y = np.asarray(read_path[:,-1])

    y = y.reshape(y.shape[0], 1) #label必须是矩阵，不可以是vector

    X = X[start:stop, :]
    y = y[start:stop, :]

    if mode == 'classification':  #对于classification的问题，需要把label转化到one_hot序列
        labels=np.unique(y)
        one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
        print("Labels are:", labels)
        for i in xrange(y.shape[0]):
            label = y[i]
            one_hot[i,np.where(labels==label)[0]] = 1.

        y = one_hot

    return DenseDesignMatrix(X=X, y=y)
