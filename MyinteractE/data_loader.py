import os
from _collections import defaultdict
import numpy as np
import torch


def get_id(path):
    eneity_dict = {}
    rel_dict = {}
    list = ['train.txt', 'test.txt', 'valid.txt']
    for i in list:
        with open(os.path.join(path, i), "r", encoding='utf8') as loader:
            for line in loader.readlines():
                h, r, t = line.strip().split()

                if h not in eneity_dict.keys():
                    eneity_dict[h] = len(eneity_dict)

                if r not in rel_dict.keys():
                    rel_dict[r] = len(rel_dict)

                if t not in eneity_dict.keys():
                    eneity_dict[t] = len(eneity_dict)

    rel_dict.update({r + '_reverse': idx + len(rel_dict) for idx, r in enumerate(rel_dict)})

    return eneity_dict, rel_dict


def data_process(path, eneity_dict, rel_dict):
    rel_num = len(rel_dict) // 2
    data = defaultdict(list)
    make_marrix = defaultdict(set)
    data_list = ['train.txt', 'test.txt', 'valid.txt']
    for i in data_list:
        with open(os.path.join(path, i), "r", encoding='utf8') as loader:
            for line in loader.readlines():
                h, r, t = line.strip().split()
                h, r, t = eneity_dict[h], rel_dict[r], eneity_dict[t]
                data[i].append((h, r, t))

                if i == 'train.txt':
                    make_marrix[(h, r)].add(t)
                    make_marrix[(t, r + rel_num)].add(h)

    data = dict(data)
    make_train_marrix = {k: list(v) for k, v in make_marrix.items()}

    for i in ['test.txt', 'valid.txt']:
        for h, r, t in data[i]:
            make_marrix[(h, r)].add(t)
            make_marrix[(t, r + rel_num)].add(h)

    make_all_marrix = {k: list(v) for k, v in make_marrix.items()}
    return data, make_train_marrix, make_all_marrix


def data_train_process(data, make_train_marrix, eneity_len, rel_len, negtive_num, label_smooth):
    train_data = []
    triples_label = []
    train_label = []
    for h, r, t in data['train.txt']:
        label = make_train_marrix[(h, r)]

        # 生成标签1,0,0,0，...，0
        triple_label = [1] + [0] * negtive_num
        triple_label = [(1.0 - label_smooth) * i + (1.0 / eneity_len) for i in triple_label]
        triples_label.append(triple_label)

        # 生成训练数据中的label：正样本，负样本，负样本，负样本，负样本，负样本，...，负样本
        entities = np.arange(eneity_len, dtype=np.int32)
        postive = np.array([t])
        mask = np.ones([eneity_len], dtype=np.bool)
        mask[label] = 0
        temp = np.random.choice(entities[mask], negtive_num, replace=False)
        train_label.append(np.concatenate((postive, temp)))

        # 生成训练集，h,r,t
        train_data.append((h, r))

        # 生成反的数据
        r_inv = r + rel_len
        label_inv = make_train_marrix[(t, r_inv)]

        # 生成标签1,0,0,0，...，0
        triple_label_inv = [1] + [0] * negtive_num
        triple_label_inv = [(1.0 - label_smooth) * i + (1.0 / eneity_len) for i in triple_label_inv]
        triples_label.append(triple_label_inv)

        # 生成训练数据中的label：正样本，负样本，负样本，负样本，负样本，负样本，...，负样本
        entities_inv = np.arange(eneity_len, dtype=np.int32)
        postive_inv = np.array([h])
        mask_inv = np.ones([eneity_len], dtype=np.bool)
        mask_inv[label_inv] = 0
        temp_inv = np.random.choice(entities_inv[mask_inv], negtive_num, replace=False)
        train_label.append(np.concatenate((postive_inv, temp_inv)))

        # 生成训练集，h,r,t
        train_data.append((t, r_inv))

    train_data = torch.LongTensor(train_data)
    triples_label = torch.FloatTensor(triples_label)
    train_label = torch.tensor(train_label)
    return train_data, triples_label, train_label

def data_test_process(data, make_all_marrix, eneity_len, rel_len):
    test_data = []
    test_label1 = []
    test_label2 = []
    for h, r, t in data['test.txt']:
        r_inv = r + rel_len
        test_data.append((h, r, r_inv, t))

        label1 = make_all_marrix[(h, r)]
        eneities1 = np.zeros([eneity_len], dtype=np.int32)
        for i in label1:
            eneities1[i] = 1
        test_label1.append(eneities1)

        label2 = make_all_marrix[(t, r_inv)]
        eneities2 = np.zeros([eneity_len], dtype=np.int32)
        for i in label2:
            eneities2[i] = 1
        test_label2.append(eneities2)

    test_data = torch.LongTensor(test_data)
    test_label1 = torch.LongTensor(test_label1)
    test_label2 = torch.LongTensor(test_label2)
    return test_data, test_label1, test_label2


def get_chequer(matrix_w, matrix_h, perm):
    dim = matrix_w * matrix_h
    ent_perm = np.int32([np.random.permutation(dim) for _ in range(perm)])
    rel_perm = np.int32([np.random.permutation(dim) for _ in range(perm)])

    comb_idx = []
    for k in range(perm):
        temp = []
        ent_idx, rel_idx = 0, 0

        for i in range(matrix_h):
            for j in range(matrix_w):
                if k % 2 == 0:
                    if i % 2 == 0:
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                        temp.append(rel_perm[k, rel_idx] + dim)
                        rel_idx += 1
                    else:
                        temp.append(rel_perm[k, rel_idx] + dim)
                        rel_idx += 1
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                else:
                    if i % 2 == 0:
                        temp.append(rel_perm[k, rel_idx] + dim)
                        rel_idx += 1
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                    else:
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                        temp.append(rel_perm[k, rel_idx] + dim)
                        rel_idx += 1
        comb_idx.append(temp)
    comb_idx = torch.LongTensor(comb_idx)
    return comb_idx
