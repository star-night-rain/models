import torch
import torch.nn as nn


class GraphSage(nn.Module):

    def __init__(self, num_classes, encoder):
        super(GraphSage, self).__init__()
        self.encoder = encoder

        self.weight = nn.Parameter(
            torch.FloatTensor(encoder.output_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        # 输出层
        output = torch.mm(embeds, self.weight)
        return output
