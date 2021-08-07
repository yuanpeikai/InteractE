import argparse
import torch
from data_loader import get_id, data_process, data_train_process, data_test_process, get_chequer
import os
from MyDataSet import My_Train_DataSet, My_Test_DataSet
import torch.utils.data as Data
from InteractE import InteractE
import torch.optim as optim
import numpy as np
from evaluation import ranking_and_hits

parser = argparse.ArgumentParser(description="Parser For Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", type=str, default='FB15k-237')
parser.add_argument("--url", type=str, default='./data')
parser.add_argument("--matrix_w", type=int, default=10)
parser.add_argument("--matrix_h", type=int, default=20)
parser.add_argument("--negtive_num", type=int, default=1000)
parser.add_argument("--label_smooth", type=float, default=0.1)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--perm", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--ker_size", type=int, default=9)
parser.add_argument("--num_ker", type=int, default=96)

args = parser.parse_args()

# 判断cuda可不可以使用
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# 隐藏层的维度
dim = args.matrix_w * args.matrix_h

print("生成id")
path = os.path.join(args.url, args.dataset)
eneity_dict = {}
rel_dict = {}
eneity_dict, rel_dict = get_id(path)
print("共有实体：{}个  关系：{}个".format(len(eneity_dict), len(rel_dict) // 2))

print("处理数据集")
data, make_train_marrix, make_all_marrix = data_process(path, eneity_dict, rel_dict)
print("数据集处理完成")
print("准备训练数据")
train_data, triples_label, train_label = data_train_process(data, make_train_marrix, len(eneity_dict),
                                                            len(rel_dict) // 2, args.negtive_num, args.label_smooth)
train_data = train_data.to(device)
train_label = train_label.to(device)
triples_label = triples_label.to(device)
train_data_batch = Data.DataLoader(dataset=My_Train_DataSet(train_data, triples_label, train_label),
                                   batch_size=args.train_batch_size, shuffle=True)
print("训练数据加载完成")
print("准备测试数据")

test_data, test_label1, test_label2 = data_test_process(data, make_all_marrix, len(eneity_dict), len(rel_dict) // 2)
test_data = test_data.to(device)
test_label1 = test_label1.to(device)
test_label2 = test_label2.to(device)

test_data_batch = Data.DataLoader(dataset=My_Test_DataSet(test_data, test_label1, test_label2),
                                  batch_size=args.test_batch_size, shuffle=True)
print("测试数据加载完成")

# 生成chequer随机序列
chequer_perm = get_chequer(args.matrix_w, args.matrix_h, args.perm)

# 加载模型

model = InteractE(len(eneity_dict), len(rel_dict), args.perm, args.matrix_w, args.matrix_h, args.num_ker,
                  args.ker_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

for epoch in range(args.epochs):
    model.train()
    loss_all = []
    for index, data in enumerate(train_data_batch):
        optimizer.zero_grad()
        train_data, triples_label, train_label = data
        h, r = train_data[:, 0], train_data[:, 1]
        # 128,1001
        pred = model(h, r, train_label, chequer_perm, args.perm, args.matrix_w, args.matrix_h, args.ker_size)
        loss = model.loss(pred, triples_label)
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()

    print("第{}次训练的loss:{}".format(epoch + 1, np.mean(loss_all)))

    model.eval()
    with torch.no_grad():
        if((epoch+1)%5==0):
            print("第{}训练的测试结果".format(epoch+1))
            ranking_and_hits(model, test_data_batch, chequer_perm, args.perm, args.matrix_w, args.matrix_h, args.ker_size,device)
