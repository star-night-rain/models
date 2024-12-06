import torch
from argparse import ArgumentParser


def get_args():
    # 参数解析器
    parser = ArgumentParser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练设置
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # 模型设置
    parser.add_argument('--input_dim', type=int, default=1433)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bias', type=bool, default=False)

    args = parser.parse_args()
    return args
