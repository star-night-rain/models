import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def load_data(seed):
    np.random.seed(seed)

    # pandas读取文件时会默认把当一行当初表头行
    content = pd.read_csv('../cora/cora.content', sep='\t', header=None)
    cities = pd.read_csv('../cora/cora.cites', sep='\t', header=None)

    labels = content.iloc[:, -1]

    # 将字符串标签转换成独热编码
    labels = pd.get_dummies(labels)

    labels = np.array(labels)
    labels = np.where(labels)[1]
    labels = labels.reshape(-1, 1)

    x = content.iloc[:, 1:-1]
    x = np.array(x)

    # 获取行索引
    content_idx = list(content.index)
    paper_idx = list(content.iloc[:, 0])
    mp = dict(zip(paper_idx, content_idx))

    n = content.shape[0]
    # 生成带默认值的字典
    adj_lists = defaultdict(set)
    for i, j in zip(cities[0], cities[1]):
        u = mp[i]
        v = mp[j]
        adj_lists[u].add(v)
        adj_lists[v].add(u)

    random_index = np.random.permutation(n)

    train_index = list(random_index[1500:])
    valid_index = random_index[1000:1500]
    test_index = random_index[:1000]

    return x, labels, adj_lists, train_index, valid_index, test_index
