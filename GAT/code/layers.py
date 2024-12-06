import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GAT


class GraphAttentionLayer(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 alpha,
                 dropout,
                 bias=True,
                 concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(2 * output_dim, 1))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        std = 1 / math.sqrt(self.w.size(1))

        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    # torch.mm()仅支持矩阵乘法
    # torch.matmul()还支持广播机制
    def forward(self, h, adj):
        wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(wh)

        # 为什么不直接使用0:softmax不会得到概率为0的分布
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # 防止聚合邻居特征的时候过拟合
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, wh)

        if self.bias is not None:
            h_prime = h_prime + self.bias

        # 增强非线性表达能力
        if self.concat:
            h_prime = F.elu(h_prime)
        return h_prime

    # 巧妙的实现
    def _prepare_attentional_mechanism_input(self, wh):
        wh1 = torch.matmul(wh, self.a[:self.output_dim, :])
        wh2 = torch.matmul(wh, self.a[self.output_dim:, :])
        # 广播加法
        e = wh1 + wh2.T
        return self.leaky_relu(e)


class SparseGraphAttentionLayer(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 alpha,
                 dropout,
                 bias=False,
                 concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.w = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(1, 2 * output_dim))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        std = 1 / math.sqrt(self.w.size(1))

        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, x, adj):
        device = x.device
        n = x.size(0)
        # 2*|E|
        edge = adj.nonzero().t()

        # |v|*output_dim
        h = torch.mm(x, self.w)

        # (n,2*output_dim) -> (2*output_dim,n)
        edge_h = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1).t()
        # 为什么要取负数:避免过度依赖注意力权重比较大的邻居
        # squeeze()去除所有大小为1的维度
        # GAT
        edge_e = torch.exp(-self.leaky_relu(self.a.mm(edge_h).squeeze()))
        # GATv2
        # edge_e = torch.exp(self.a.mm(-self.leaky_relu(edge_h)).squeeze())

        e_row_sum = self._special_mm(edge, edge_e, torch.Size([n, n]),
                                     torch.ones((n, 1), device=device))

        edge_e = F.dropout(edge_e, self.dropout, training=self.training)

        h_prime = self._special_mm(edge, edge_e, torch.Size([n, n]), h)

        # Normalize node features
        h_prime = h_prime.div(e_row_sum)

        if self.bias is not None:
            h_prime = h_prime + self.bias

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # 稀疏矩阵和向量相乘
    # indices:(2,n)
    # values: (n)
    # shape: (n,n)
    # b: (n,1)
    def _special_mm(self, indices, values, shape, b):
        return torch.sparse_coo_tensor(indices, values, shape).matmul(b)
