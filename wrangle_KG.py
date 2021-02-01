from __future__ import print_function
from os.path import join
import json

import argparse
import datetime
import json
import urllib
import pickle
import os
import numpy as np
import operator
import sys

"""这个文件在预处理阶段被执行，即运行sh preprocess.sh时会运行此程序"""
rdm = np.random.RandomState(234234)

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]  # 第二个参数是数据集的名字
else:
    dataset_name = 'FB15k-237'
    #dataset_name = 'FB15k'
    #dataset_name = 'yago'
    #dataset_name = 'WN18RR'

print('Processing dataset {0}'.format(dataset_name))

rdm = np.random.RandomState(2342423)
base_path = 'data/{0}/'.format(dataset_name)
files = ['train.txt', 'valid.txt', 'test.txt']

data = []  # data是存放训练数据、验证数据和测试数据的列表，三元组们都放在一起了
for p in files:
    with open(join(base_path, p)) as f:
        data = f.readlines() + data  # data是由数据集每一行的三元组字符串作为元素（包含换行符\n）构成的列表
        # 例如['e1    r1  e3\n', 'e4    r5  e2\n', ....]

label_graph = {}  # 主要是创建以下三种数据。数据存放格式是怎样的看下面程序
train_graph = {}
test_cases = {}
for p in files:         # 注意test_cases和train_graph是分数据类型的，即训练、验证、测试
    test_cases[p] = []  # test_cases结构是{'train.txt': [], 'valid.txt': [], 'test.txt': []}
    train_graph[p] = {}  # train_graph结构是{'train.txt': {}, 'valid.txt': {}, 'test.txt': {}}


for p in files:  # 对于此数据集的训练或验证或测试数据
    with open(join(base_path, p)) as f:
        for i, line in enumerate(f):  # i为每一行序号，line为一行三元组组成的字符串
            e1, rel, e2 = line.split('\t')  # 以'\t'字符分割开每个元素
            e1 = e1.strip()  # 去除左右空白字符
            e2 = e2.strip()  # 去除左右空白字符
            rel = rel.strip()  # 去除此关系字符串的左右空白字符
            rel_reverse = rel + '_reverse'  # 论文中原来用到了关系的逆。关系的逆可以增强知识图谱中的连接特征多样性

            # data
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
                                               # 注意label_graph中并没有分是训练还是验证还是测试，而是整个数据集中的三元组都放在一起了。
            if (e1, rel) not in label_graph:   # label_graph是字典{(e1, rel): (), ...}，以元组(头实体字符串, 关系字符串)为键，空集合为值
                label_graph[(e1, rel)] = set()  # e1是头实体字符串，rel是关系字符串

            if (e2,  rel_reverse) not in label_graph:  # 把(尾实体字符串, 关系的逆字符串)元组作为键，空集合作为值
                label_graph[(e2, rel_reverse)] = set()

            if (e1,  rel) not in train_graph[p]:  # 把(头实体字符串, 关系字符串)作为train_graph{'train.txt': {(e1, rel): ()}}字典中的字典的键
                train_graph[p][(e1, rel)] = set()
            if (e2, rel_reverse) not in train_graph[p]:  # 把(尾实体字符串, 关系的逆字符串)作为train_graph{'train.txt': {(e2, rel_reverse): ()}}字典中的字典的键
                train_graph[p][(e2, rel_reverse)] = set()

            # labels
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, Mike)
            label_graph[(e1, rel)].add(e2)  # 在label_graph是字典{(e1, rel): (e2), ...}中某个头关系二元组键对应的值的集合中加入尾实体字符串

            label_graph[(e2, rel_reverse)].add(e1)  # 在label_graph是字典{(e2, rel_reverse): (e1), ...}中某个关系尾实体二元组键对应的值的集合中加入头实体字符串

            # test cases
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            test_cases[p].append([e1, rel, e2])  # 在test_cases字典{'train.txt': [], 'valid.txt': [], 'test.txt': []}中，例如'train.txt'键的值列表[]中加入[e1, rel, e2]三元组为列表
                                                 # 注意是添加进去的元素是列表[e1, rel, e2]，而不是三个独立的元素

            # data
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)
            train_graph[p][(e1, rel)].add(e2)  # 把头和关系对应的尾实体，加入到train_graph{'train.txt': {(e1, rel): ()}}字典中的字典的键的值，即集合中
            train_graph[p][(e2, rel_reverse)].add(e1)  # 把尾和关系的逆对应的头实体，加入到train_graph{'train.txt': {(e2, rel_reverse): ()}}字典中的字典的键的值，即集合中

# 所以总结来说，得到了以下三种数据： label_graph没区分数据集，train_graph有区分数据集，除此之外，label_graph和train_graph其实是一样的。
# label_graph为{(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): (e1, ...), ...}
# train_graph为{'train.txt': {(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): 集合(e1, ...)}, 'valid.txt': {}, 'test.txt': {}}
# test_cases为{'train.txt': [[e1, rel, e2]，[e1, rel, e2], ...], 'valid.txt': [], 'test.txt': []}  注意test_cases中的数据并没有关系的逆

def write_training_graph(cases, graph, path):  # graph是train_graph['train.txt']或label_graph
    with open(path, 'w') as f:  # 当graph 为 train_graph['train.txt']时，即字典{(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): 集合(e1, ...)}
        n = len(graph)    # 其实json文件中的内容本质上就是字典
        for i, key in enumerate(graph):  # 对每一对头关系二元组(e1, rel)
            e1, rel = key  # 取出头实体和关系
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)

            # (John, fatherOf) -> Tom
            # (John, fatherOf_reverse, Mike) 
            entities1 = " ".join(list(graph[key]))  # 将当前头关系二元组所对应的尾实体集合(e2, e5, ...)取出来，变成列表，然后各个尾实体名字符串之间用空格分开

            data_point = {}
            data_point['e1'] = e1  # 将头实体名字字符串写入到'e1'为键的值中
            data_point['e2'] = 'None'  # 'e2'键的值为None
            data_point['rel'] = rel  # 'rel'键的值为关系名字字符串
            data_point['rel_eval'] = 'None'  #
            data_point['e2_multi1'] = entities1  # 'e2_multi1'键的值为字符串，例"/m/026wp /m/06cx9"
            data_point['e2_multi2'] = "None"

            f.write(json.dumps(data_point) + '\n')  # 每行都是一个字典，然后是换行符到下一行
            # 例如第一行{"e1": "/m/027rn", "e2": "None", "rel": "/location/country/form_of_government", "rel_eval": "None", "e2_multi1": "/m/026wp /m/06cx9", "e2_multi2": "None"}

# test_cases['valid.txt']或者test_cases['test.txt']都为[[e1, rel, e2]，[e1, rel, e2], ...]，里面的列表元素都是三元组。注意test_cases中的数据并没有关系的逆
# label_graph为字典{(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): (e1, ...), ...}
def write_evaluation_graph(cases, graph, path):  # cases是test_cases['valid.txt']，graph是label_graph
    with open(path, 'w') as f:
        n = len(cases)
        n1 = 0
        n2 = 0
        for i, (e1, rel, e2) in enumerate(cases):  # 对于验证或者测试数据中的每一条三元组
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel+'_reverse'  # 因为test_cases中并没有关系的逆，因此要构造
            entities1 = " ".join(list(graph[(e1, rel)]))  # 根据头关系二元组(e1, rel)键，拿出值集合(e2, e5, ...)。形成字符串，以空格隔开字符串
            entities2 = " ".join(list(graph[(e2, rel_reverse)]))  # 根据尾关系的逆二元组(e2, rel_reverse)键，拿出值集合(e1, e3, ...)。形成字符串，以空格隔开字符串
            # entities1是尾实体们"e2 e5 e6"的字符串  # entities2是头实体们"e1 e3 e8"的字符串

            n1 += len(entities1.split(' '))  # 统计尾实体个数，注意，这里的尾实体个数是有重复的
            n2 += len(entities2.split(' '))  # 统计头实体个数，注意，这里的尾实体个数是有重复的


            data_point = {}
            data_point['e1'] = e1  # 将头实体名字字符串写入到'e1'为键的值中
            data_point['e2'] = e2  # 将尾实体名字字符串写入到'e2'为键的值中
            data_point['rel'] = rel  # 'rel'键的值为关系名字字符串
            data_point['rel_eval'] = rel_reverse  # 'rel_eval'键的值为关系的逆的名字字符串
            data_point['e2_multi1'] = entities1  # e2_multi1指的是某个头实体和某个关系 对应的尾实体们的字符串(以空格隔开)
            data_point['e2_multi2'] = entities2  # e2_multi2指的是某个某个尾实体和某个关系的逆 对应的头实体们的字符串(以空格隔开)
            # "e2_multi1": "/m/01q99h /m/013w2r /m/02dw1_ /m/07m4c /m/07mvp"
            # "e2_multi2": "/m/05148p4 /m/0342h /m/018vs /m/01hww_ /m/04rzd /m/0l14qv /m/03qjg /m/0l14md /m/0l14j_ /m/02hnl /m/028tv0 /m/07y_7"

            f.write(json.dumps(data_point) + '\n')


# label_graph为{(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): (e1, ...), ...}
# train_graph为{'train.txt': {(e1, rel): 集合(e2, e5, ...), (e2, rel_reverse): 集合(e1, ...)}, 'valid.txt': {}, 'test.txt': {}}
# test_cases为{'train.txt': [[e1, rel, e2]，[e1, rel, e2], ...], 'valid.txt': [], 'test.txt': []}  注意test_cases中的数据并没有关系的逆

all_cases = test_cases['train.txt'] + test_cases['valid.txt'] + test_cases['test.txt']  # 将所有三元组都混在一起，注意还是以列表为元素
write_training_graph(test_cases['train.txt'], train_graph['train.txt'], 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name))
# 注意是调用write_training_graph函数。将test_cases['train.txt']和train_graph['train.txt']的数据，即依据字典写入json文件中

write_evaluation_graph(test_cases['valid.txt'], label_graph, join('data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)))
# 注意是调用write_evaluation_graph函数。将test_cases['valid.txt']和label_graph的训练数据写入json文件中

write_evaluation_graph(test_cases['test.txt'], label_graph, 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name))
# 注意是调用write_evaluation_graph函数。将test_cases['test.txt']和label_graph的训练数据写入json文件中

write_training_graph(all_cases, label_graph, 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name))
# 注意这里是all_cases，即[[e1, rel, e2]，[e1, rel, e2], ...]，其中的元素是列表，有可能来自训练数据或者验证数据或者测试数据
