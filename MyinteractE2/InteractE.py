import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class InteractE(nn.Module):
    def __init__(self, p):
        # ent_num, rel_num, perm, matrix_w, matrix_h, num_ker, ker_size
        super(InteractE, self).__init__()
        self.ent_embed = nn.Embedding(p.ent_len, p.dim)
        self.rel_embed = nn.Embedding(p.rel_len*2, p.dim)
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        self.bceloss = nn.BCELoss()

        self.input_drop = nn.Dropout(0.2)
        self.feature_map_drop = nn.Dropout2d(0.5)
        self.hidden_drop = nn.Dropout(0.5)

        self.bn0 = nn.BatchNorm2d(p.perm)
        self.bn1 = nn.BatchNorm2d(p.perm * p.num_ker)
        self.bn2 = nn.BatchNorm1d(p.dim)

        self.linear_size = p.perm * p.num_ker * 2 * p.matrix_w * p.matrix_h

        self.fc = nn.Linear(self.linear_size, p.dim)

        self.register_parameter('conv_filter',
                                Parameter(torch.zeros([p.num_ker, 1, p.ker_size, p.ker_size])))  # 96,1,9,9
        self.register_parameter('bias', Parameter(torch.zeros([p.ent_len])))  # 14541
        nn.init.xavier_uniform_(self.conv_filter)

    def loss(self, pred, triples_label):
        loss = self.bceloss(pred, triples_label)
        return loss

    def circular_padding_chw(self, x, padding):
        upper_pad = x[..., -padding:, :]  # batch_size,perm,4,20
        lower_pad = x[..., :padding, :]  # batch_size,perm,4,20
        x = torch.cat([upper_pad, x, lower_pad], dim=2)  # batch_size,perm,28,20

        left_pad = x[..., -padding:]  # batch_size,perm,28,4
        right_pad = x[..., :padding]  # batch_size,perm,28,4
        x = torch.cat([left_pad, x, right_pad], dim=3)  # batch_size,perm,28,28
        return x  # batch_size,perm,28,28

    def forward(self, p, h, r):
        h_embed = self.ent_embed(h)  # batch_size,dim
        r_embed = self.rel_embed(r)  # batch_size,dim
        comb_embed = torch.cat([h_embed, r_embed], dim=1)  # batch_size,2*dim
        chequer_perm = comb_embed[:, p.chequer_perm]  # batch_size,1,2*dim
        stack_inp = chequer_perm.reshape([-1, p.perm, 2 * p.matrix_w, p.matrix_h])  # batch_size,perm,2*w,h
        stack_inp = self.bn0(stack_inp)
        x = self.input_drop(stack_inp)

        # 处理输入矩阵，变成循环卷积的形状,卷积核一定要为单数，不然无法执行
        x = self.circular_padding_chw(x, p.ker_size // 2)  # batch_size,perm,28,28
        # 卷积操作
        filter = self.conv_filter.repeat(p.perm, 1, 1, 1)  # perm*96,1,9,9
        x = F.conv2d(x, filter, padding=0, groups=p.perm)  # [128, perm*96, 20, 20]

        x = self.bn1(x)  # [128, 96, 20, 20]
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.linear_size)  # 128,96*20*20
        x = self.fc(x)  # 128,200
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))  # batcch_size,ent_len
        x += self.bias.expand_as(x)
        return torch.sigmoid(x)
