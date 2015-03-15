#-*- coding:utf-8 -*-

YAML is a human-readable dataset serialization scripting language,for defining an experimental configuration


YAML不可以用双引号，不可以用\t，只能用空格
每行语句要注意空格和逗号，尤其是冒口以后

#构造YAML !pkl/!import/!obj：
!obj:yaml_tutorial.autoencoder.AutoEncoder {
   "nvis": &nvis 784, #在一个yaml文件里，可以用&定义一个变量/类的实例，再用的时候就是*，有点像指针
   "nhid": *nvis,
   "iscale": 0.2,
   "activation_fn": !import 'pylearn2.expr.nnet.sigmoid_numpy',
   "params": !pkl: 'example3_weights.pkl'}

# !obj第一是import沿途所有package,他们都必须在pythopath里 + 实例化一个最后一个类的对象
# !import也可以一路import所有的package，但是它不实例化一个类
# !pkl可以读取用pickle.dump存的东西，作用和Pickle.load一样


#调用YAML：
fp = open('example1.yaml')
model = yaml_parse.load(fp) #运行YAML的内容：import+实例化
print model #__str__属性是Print专属的
model.save('example3_weights.pkl')
fp.close()

#实际是这样调用：
train.py dae.yaml  #scripts的train.py要放在path理


plot_monitor.py: lets you plot channels recorded in the model’s monitor
print_monitor.py: prints out the final value of each channel in the monitor.
summarize_model.py: prints out a few statistics of the model’s parameters
show_weights.py: displays a visualization of the weights of your model’s first layer.


pylearn2的三步走原则： dataset+model+training algo

!obj:pylearn2.train.Train {

    "dataset": !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix &dataset {
        "X" : !obj:numpy.random.normal { 'size':[5,3] },
    },

    "model": !obj:pylearn2.models.autoencoder.DenoisingAutoencoder {
        "nvis" : 3,
        "nhid" : 4,
        "irange" : 0.05,
        "corruptor": !obj:pylearn2.corruption.BinomialCorruptor {
            "corruption_level": 0.5,
        },
        "act_enc": "tanh",
        "act_dec": null,    # Linear activation on the decoder side.
    },

    "algorithm": !obj:pylearn2.training_algorithms.sgd.SGD {
        "learning_rate" : 1e-3,
        "batch_size" : 5,
        "monitoring_dataset" : *dataset,
        "cost" : !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {},
        "termination_criterion" : !obj:pylearn2.termination_criteria.EpochCounter {
            "max_epochs": 1,
        },
    },

    "save_path": "./garbage.pkl"
}