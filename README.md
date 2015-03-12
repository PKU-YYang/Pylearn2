Pylearn2 in practice
====================

See [http://fastml.com/pylearn2-in-practice/](http://fastml.com/pylearn2-in-practice/) for description.


Classification:
python predict_csv.py softmax_regression_best.pkl adult/test.csv output.txt -L
test.csv因为第一列是label，所以这里必须-L
这个函数会同时输出prob_matrix和label

Regression:
python predict_csv.py wine.pkl wine_test.csv output.csv -H -P regression -T float
test.csv不可以有label/value, 如果有带表头要-H