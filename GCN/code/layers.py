import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class GraphConvolution(Module):

    def __init__(self, input_dim, output_dim, bias=True):
        # super()的第一个参数必须是子类名称
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 变成模型参数
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            # 向模型注册参数
            self.register_parameter('bias', None)

        self.reset_parameters()

    # TODO 初始化的时候什么情况下考虑均匀分布或高斯分布?
    # TODO bias如何初始化
    def reset_parameters(self):
        # 原始初始化方法
        stdv = 1 / math.sqrt(self.weight.size(1))
        # 原始方法为83.80%,下面的方法加上邻接矩阵归一化（按特征处理）为84.60%
        # stdv = 1 / self.weight.size(1)
        # 1.414是ReLU的建议增益值
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        output = torch.mm(adj, x)
        if self.bias is not None:
            output = output + self.bias
        return output
