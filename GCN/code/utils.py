import numpy as np
import pandas as pd
import torch


def accuracy(output, labels):
    # max(1)表示计算每行的最大值，并返回最大值和对应的索引
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double().sum()
    return correct / len(labels)


# 归一化邻接矩阵
def normalize_adj(mx):
    # 按行求和
    row_sum = np.array(mx.sum(axis=1), dtype=np.float32)
    # 计算每个元素的倒数
    row_inv = np.power(row_sum, -0.5)
    # 防止除以零
    row_inv[np.isinf(row_inv)] = 0
    # 创建对角矩阵，对角元素为row_inv
    row_mat_inv = np.diag(row_inv)
    return mx.dot(row_mat_inv).transpose().dot(row_mat_inv)


# 归一化顶点特征（除以顶点度数）
def normalize_feature(mx):
    # 按行求和
    row_sum = np.array(mx.sum(axis=1), dtype=np.float32)
    # 计算每个元素的倒数
    row_inv = np.power(row_sum, -1)
    # 防止除以零
    row_inv[np.isinf(row_inv)] = 0
    # 创建对角矩阵，对角元素为row_inv
    row_mat_inv = np.diag(row_inv)
    mx = np.dot(row_mat_inv, mx)
    return mx


def load_data():
    # pandas读取文件时会默认把当一行当初表头行
    content = pd.read_csv('../cora/cora.content', sep='\t', header=None)
    cities = pd.read_csv('../cora/cora.cites', sep='\t', header=None)

    # 获取指定位置的数据
    labels = content.iloc[:, -1]
    # 将字符串标签转换成独热编码
    labels = pd.get_dummies(labels)

    labels = np.array(labels)
    # np.where()返回满足条件（非零）元素的行、列索引
    labels = torch.LongTensor(np.where(labels)[1])

    features = content.iloc[:, 1:-1]
    # TODO 特征也需要做归一化吗？
    features = normalize_feature(features)
    features = np.array(features)
    features = torch.FloatTensor(features)

    # 获取行索引
    content_idx = list(content.index)
    paper_idx = list(content.iloc[:, 0])
    mp = dict(zip(paper_idx, content_idx))

    n = content.shape[0]
    adj = np.zeros((n, n))
    for i, j in zip(cities[0], cities[1]):
        if i not in mp or j not in mp:
            continue
        x = mp[i]
        y = mp[j]
        adj[x, y] = 1
        adj[y, x] = 1
    # 添加自环
    eye = np.eye(n)
    adj = adj + eye

    # 邻接矩阵归一化
    adj = normalize_adj(adj)
    adj = torch.FloatTensor(adj)

    train_index = range(140)
    valid_index = range(200, 500)
    test_index = range(500, 1500)

    train_index = torch.LongTensor(train_index)
    valid_index = torch.LongTensor(valid_index)
    test_index = torch.LongTensor(test_index)

    return features, adj, labels, train_index, valid_index, test_index
