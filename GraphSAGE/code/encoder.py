import torch
import torch.nn as nn
import torch.nn.functional as F
'''根据自身和邻居的信息，更新自身的信息'''
from torch_geometric.nn import GraphSAGE


class Encoder(nn.Module):

    def __init__(self,
                 features,
                 input_dim,
                 output_dim,
                 adj_lists,
                 aggregator,
                 base_model=None,
                 num_sample=10,
                 self_loop=True):
        super(Encoder, self).__init__()
        self.features = features
        self.output_dim = output_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model
        self.num_sample = num_sample
        self.self_loop = self_loop

        self.weight = nn.Parameter(
            torch.FloatTensor(input_dim * 2 if self.self_loop else input_dim,
                              output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        neighbor_features = self.aggregator.forward(
            nodes, [self.adj_lists[int(node)] for node in nodes],
            self.num_sample)

        if self.self_loop:
            self_features = self.features(torch.LongTensor(nodes))
            combined_features = torch.cat([self_features, neighbor_features],
                                          dim=1)
        else:
            combined_features = neighbor_features

        output = F.relu(combined_features.mm(self.weight))

        return output
