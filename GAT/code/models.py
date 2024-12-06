import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SparseGraphAttentionLayer


class GATBase(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, alpha,
                 dropout, num_layers, bias, attention_layer):
        super(GATBase, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers == 2:
            self.attentions = nn.ModuleList()
            for _ in range(num_heads):
                self.attentions.append(
                    attention_layer(input_dim,
                                    hidden_dim,
                                    alpha,
                                    dropout,
                                    bias=bias,
                                    concat=True))
            self.out_att = attention_layer(num_heads * hidden_dim,
                                           output_dim,
                                           alpha,
                                           dropout,
                                           bias=bias,
                                           concat=False)
        else:
            # 创建多层注意力层
            self.attentions = nn.ModuleList()
            attention = nn.ModuleList()
            for _ in range(num_heads):
                attention.append(
                    attention_layer(input_dim,
                                    hidden_dim,
                                    alpha,
                                    dropout,
                                    bias=bias,
                                    concat=True))
            self.attentions.append(attention)

            for _ in range(num_layers - 2):
                attention = nn.ModuleList()
                for _ in range(num_heads):
                    attention.append(
                        attention_layer(num_heads * hidden_dim,
                                        hidden_dim,
                                        alpha,
                                        dropout,
                                        bias=bias,
                                        concat=True))
                self.attentions.append(attention)

            self.attentions.append(
                attention_layer(num_heads * hidden_dim,
                                output_dim,
                                alpha,
                                dropout,
                                bias=bias,
                                concat=False))

    def forward(self, x, adj):
        if self.num_layers == 2:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=1)
        else:
            for layer in range(len(self.attentions) - 1):
                # 防止计算注意力系数的时候过拟合
                x = F.dropout(x, self.dropout, training=self.training)
                x = torch.cat([att(x, adj) for att in self.attentions[layer]],
                              dim=1)
            # 防止输出层过拟合
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.attentions[-1](x, adj))
            return F.log_softmax(x, dim=1)


class GAT(GATBase):

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, alpha,
                 dropout, num_layers, bias):
        # 调用 GATBase，并传入 GraphAttentionLayer
        super(GAT, self).__init__(input_dim, hidden_dim, output_dim, num_heads,
                                  alpha, dropout, num_layers, bias,
                                  GraphAttentionLayer)


class SpGAT(GATBase):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads,
        alpha,
        dropout,
        num_layers,
        bias,
    ):
        # 调用 GATBase，并传入 SparseGraphAttentionLayer
        super(SpGAT, self).__init__(input_dim, hidden_dim, output_dim,
                                    num_heads, alpha, dropout, num_layers,
                                    bias, SparseGraphAttentionLayer)
