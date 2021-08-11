import argparse
import torch
from data_loader import get_id, data_process, data_train_process, data_test_process, get_chequer, data_valid_process
import os
from MyDataSet import My_Train_DataSet, My_Test_DataSet
import torch.utils.data as Data
import time
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
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--ker_size", type=int, default=9)
parser.add_argument("--num_ker", type=int, default=96)

args = parser.parse_args()


class Main():
    def __init__(self, args):
        # 判断cuda可不可以使用
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.matrix_w = args.matrix_w
        self.matrix_h = args.matrix_h
        self.dim = args.matrix_w * args.matrix_h
        self.perm = args.perm
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.label_smooth = args.label_smooth
        self.num_ker = args.num_ker
        self.ker_size = args.ker_size
        self.lr = args.lr
        self.l2 = args.l2
        self.epochs = args.epochs

        self.chequer_perm = self.get_chequer_perm()
        # 隐藏层的维度
        self.dim = args.matrix_w * args.matrix_h
        print("生成id")
        path = os.path.join(args.url, args.dataset)
        self.eneity_dict = {}
        self.rel_dict = {}
        self.eneity_dict, self.rel_dict = get_id(path)
        self.ent_len, self.rel_len = len(self.eneity_dict), len(self.rel_dict) // 2
        print("共有实体：{}个  关系：{}个".format(self.ent_len, self.rel_len))
        print("处理数据集")
        self.data, self.make_train_marrix, self.make_all_marrix = data_process(path, self.eneity_dict, self.rel_dict)
        print("数据集处理完成")

    def train_data(self):
        time1 = time.time()
        print("准备训练数据")
        train_data, train_label = data_train_process(self)
        train_data_batch = Data.DataLoader(dataset=My_Train_DataSet(train_data, train_label),
                                           batch_size=self.train_batch_size, shuffle=True)
        print("训练数据加载完成")
        time2 = time.time()
        print("训练集加载时间：{}".format(time2 - time1))
        return train_data_batch

    def test_data(self):
        time1 = time.time()
        print("准备测试数据")
        test_data, test_label1, test_label2 = data_test_process(self)
        test_data_batch = Data.DataLoader(dataset=My_Test_DataSet(test_data, test_label1, test_label2),
                                          batch_size=self.test_batch_size, shuffle=True)
        print("测试数据加载完成")
        time2 = time.time()
        print("测试集加载时间：{}".format(time2 - time1))
        return test_data_batch

    def valid_data(self):
        time1 = time.time()
        print("准备验证数据")
        valid_data, valid_label1, valid_label2 = data_valid_process(self)
        valid_data_batch = Data.DataLoader(dataset=My_Test_DataSet(valid_data, valid_label1, valid_label2),
                                           batch_size=self.test_batch_size, shuffle=True)
        print("验证数据加载完成")
        time2 = time.time()
        print("验证集加载时间：{}".format(time2 - time1))
        return valid_data_batch

    def get_chequer_perm(self):
        # 生成chequer随机序列
        return get_chequer(self)

    def fit(self, train_data_batch, test_data_batch, valid_data_batch):
        # 加载模型
        model = InteractE(self).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)
        # 开始训练
        for epoch in range(self.epochs):
            model.train()
            loss_all = []
            for index, data in enumerate(train_data_batch):
                optimizer.zero_grad()
                train_data, train_label = data
                h, r = train_data[:, 0], train_data[:, 1]
                pred = model(self, h, r)
                # 标签平滑处理

                train_label = ((1.0 - self.label_smooth) * train_label) + (1.0 / train_label.size(1))

                loss = model.loss(pred, train_label)
                loss_all.append(loss.item())
                loss.backward()
                optimizer.step()
            if ((epoch + 1) % 10 == 0):
                print("第{}次训练的loss:{}".format(epoch + 1, np.mean(loss_all)))

            if ((epoch + 1) % 100 == 0):
                # 保存模型
                torch.save(model.state_dict(), '/tmp/pycharm_project_862/model/model_{}.pkl'.format(epoch + 1))

            model.eval()
            with torch.no_grad():
                if ((epoch + 1) % 10 == 0):
                    print("***************************************************************")
                    print("第{}测试集结果".format(epoch + 1))
                    ranking_and_hits(self, model, test_data_batch)
                    print("***************************************************************")
                    print("第{}验证集结果".format(epoch + 1))
                    ranking_and_hits(self, model, valid_data_batch)
                    print("***************************************************************")

main = Main(args)
train_data_batch = main.train_data()
test_data_batch = main.test_data()
valid_data_batch = main.valid_data()
main.fit(train_data_batch, test_data_batch, valid_data_batch)
