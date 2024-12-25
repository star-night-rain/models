import torch
import torch.nn as nn
import random
'''使用平均值的方式聚合邻居信息'''


class MeanAggregator(nn.Module):

    def __init__(self, features, self_loop=True):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.self_loop = self_loop

    def forward(self, nodes, neighbors, num_sample):
        # 采集邻居
        sample_neighbors = [
            # random.sample不会选择重复的邻居
            set(random.sample(neighbor, num_sample))
            if len(neighbor) >= num_sample else neighbor
            for neighbor in neighbors
        ]

        # 添加自环
        if self.self_loop:
            sample_neighbors = [
                sample_neighbor.union(set([nodes[i].item()]))
                # (index,value)
                for i, sample_neighbor in enumerate(sample_neighbors)
            ]
        # print(sample_neighbors)
        # *表示解包sample_neighbors的所有集合作为union的参数
        unique_nodes_list = list(set.union(*sample_neighbors))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # (节点数，邻居数)
        mask = torch.zeros(len(sample_neighbors), len(unique_nodes))
        row_indices = [
            i for i in range(len(sample_neighbors))
            for j in range(len(sample_neighbors[i]))
        ]
        column_indices = [
            unique_nodes[n] for sample_neighbor in sample_neighbors
            for n in sample_neighbor
        ]
        mask[row_indices, column_indices] = 1

        # keepdim=True表示求和后保持tensor的维度不变
        mask_row_sum = mask.sum(1, keepdim=True)
        # 广播机制
        mask = mask.div(mask_row_sum)

        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        output = mask.mm(embed_matrix)
        return output
