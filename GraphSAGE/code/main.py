import torch
import torch.nn as nn
from aggregator import *
from encoder import Encoder
from get_args import get_args
from utils import load_data
from models import GraphSage
from sklearn.metrics import f1_score
import numpy as np
import time


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    x, labels, adj_lists, train_index, valid_index, test_index = load_data(
        args.seed)

    # 冻结节点的初始特征
    features = nn.Embedding(args.num_nodes, args.input_dim)
    features.weight = nn.Parameter(torch.FloatTensor(x), requires_grad=False)

    agg1 = MeanAggregator(features, self_loop=True)
    enc1 = Encoder(features,
                   args.input_dim,
                   args.hidden_dim,
                   adj_lists,
                   agg1,
                   args.first_order_neighbors,
                   self_loop=True)
    # # nodes是匿名函数的参数
    agg2 = MeanAggregator(lambda nodes: enc1(nodes), self_loop=True)
    # base_model的作用是把enc1和agg1的参数也放到enc2中，从而进行梯度更新
    # 如果没有base_model，那么enc1的参数保持不变
    enc2 = Encoder(lambda nodes: enc1(nodes),
                   args.hidden_dim,
                   args.hidden_dim,
                   adj_lists,
                   agg2,
                   base_model=enc1,
                   num_sample=args.second_order_neighbors,
                   self_loop=True)

    graphsage = GraphSage(args.output_dim, enc2)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       graphsage.parameters()),
                                lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = 1e9
    best_model = None

    start_time = time.time()

    for epoch in range(1, 1 + args.epochs):
        batch_nodes = train_index[:256]
        random.shuffle(train_index)

        output = graphsage(batch_nodes)
        # (256,7) and (256,)
        # squeeze()去掉维度为1的维度
        train_loss = criterion(output,
                               torch.LongTensor(labels[batch_nodes]).squeeze())
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # valid_loss = criterion(graphsage(valid_index),
        #                        torch.LongTensor(
        #                            labels[valid_index]).squeeze()).item()
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     best_model = graphsage
        # print(
        #     f'epoch:{epoch},train loss:{train_loss.item()},valid loss:{valid_loss}'
        # )

    end_time = time.time()
    print(f'training time:{end_time-start_time:.2f}s')

    output = graphsage(test_index)
    f1 = f1_score(labels[test_index],
                  output.data.numpy().argmax(axis=1),
                  average='micro')
    print(f'f1 score:{f1*100:.2f}%')


if __name__ == '__main__':
    main()
