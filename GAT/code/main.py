from get_args import get_args
from utils import load_data, accuracy
from models import GAT, SpGAT
import torch
import torch.optim as optim


def main(sparse):
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    features, adj, labels, train_index, valid_index, test_index = load_data()
    features = features.to(args.device)
    labels = labels.to(args.device)
    adj = adj.to(args.device)
    train_index = train_index.to(args.device)
    valid_index = valid_index.to(args.device)
    test_index = test_index.to(args.device)

    if sparse:
        model = SpGAT(args.input_dim, args.hidden_dim, args.output_dim,
                      args.num_heads, args.alpha, args.dropout,
                      args.num_layers, args.bias).to(args.device)
    else:
        model = GAT(args.input_dim, args.hidden_dim, args.output_dim,
                    args.num_heads, args.alpha, args.dropout, args.num_layers,
                    args.bias).to(args.device)
    # model.load_state_dict(torch.load('../best_model.pth', weights_only=True))
    # 使用的是L2正则化
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    # 配合log_softmax()
    criterion = torch.nn.NLLLoss()

    best_model = None
    best_loss = args.epochs
    best_acc = 0
    bad_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, features, adj, labels,
                                      train_index, optimizer, criterion)
        valid_loss, valid_acc = eval(model, features, adj, labels, valid_index,
                                     criterion)
        if valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_acc = valid_acc
            bad_counter = 0
        else:
            bad_counter += 1

        print(
            f'Epoch:{epoch:03d},train loss:{train_loss:.4f}, train accuracy:{100*train_acc:.2f}%, '
            f'valid loss:{valid_loss:.4f}, valid accuracy:{100*valid_acc:.2f}%'
        )

        if bad_counter == args.patience:
            break

    # torch.save(best_model.state_dict(), '../best_model.pth')
    print(f'best valid accuracy:{100*best_acc:.2f}%')
    test_loss, test_acc = eval(best_model, features, adj, labels, test_index,
                               criterion)
    print(f'test loss:{test_loss:.4f},test accuracy:{100*test_acc:.2f}%')


def train(model, features, adj, labels, train_index, optimizer, criterion):
    # 将模型设置为训练模式
    model.train()
    output = model(features, adj)
    train_loss = criterion(output[train_index], labels[train_index])
    train_acc = accuracy(output[train_index], labels[train_index])
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss.item(), train_acc


def eval(model, features, adj, labels, data_index, criterion):
    # 将模型设置为评估模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        output = model(features, adj)
        test_loss = criterion(output[data_index], labels[data_index])
        test_acc = accuracy(output[data_index], labels[data_index])
    return test_loss.item(), test_acc


if __name__ == '__main__':
    main(sparse=True)
