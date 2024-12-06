import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvolution(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.convs.append(GraphConvolution(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, adj):
        for layer in range(len(self.convs) - 1):
            x = self.convs[layer](x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        output = self.convs[-1](x, adj)
        # log_softmax能够加快运算速度，提高数据稳定性
        output = F.log_softmax(output, dim=1)
        return output
