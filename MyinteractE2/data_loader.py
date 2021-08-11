import os
from _collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import time


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


def data_train_process(p):

    train_data = []
    train_label = torch.zeros(len(p.data['train.txt']) * 2, p.ent_len).to(p.device)
    i = 0
    for h, r, t in tqdm(p.data['train.txt']):
        r_inv = r + p.rel_len
        label = p.make_train_marrix[(h, r)]
        label_inv = p.make_train_marrix[(t, r_inv)]

        train_data.append((h, r))
        for j in label:
            train_label[i][j] = 1

        train_data.append((t, r_inv))
        for j in label_inv:
            train_label[i + 1][j] = 1
        i += 2
        # if (i == 256):
        #     break

    train_data = torch.LongTensor(train_data).to(p.device)
    return train_data, train_label


def data_test_process(p):
    test_data = []
    test_label1 = torch.zeros([len(p.data['test.txt']), p.ent_len]).to(p.device)
    test_label2 = torch.zeros([len(p.data['test.txt']), p.ent_len]).to(p.device)
    i = 0
    for h, r, t in tqdm(p.data['test.txt']):
        r_inv = r + p.rel_len
        test_data.append((h, r, r_inv, t))

        label1 = p.make_all_marrix[(h, r)]
        for j in label1:
            test_label1[i][j] = 1

        label2 = p.make_all_marrix[(t, r_inv)]
        for j in label2:
            test_label2[i][j] = 1

        i += 1

        # if (i == 256):
        #     break


    test_data = torch.LongTensor(test_data).to(p.device)
    return test_data, test_label1, test_label2


def data_valid_process(p):
    valid_data = []
    valid_label1 = torch.zeros([len(p.data['valid.txt']), p.ent_len]).to(p.device)
    valid_label2 = torch.zeros([len(p.data['valid.txt']), p.ent_len]).to(p.device)
    i = 0
    for h, r, t in tqdm(p.data['valid.txt']):
        r_inv = r + p.rel_len
        valid_data.append((h, r, r_inv, t))

        label1 = p.make_all_marrix[(h, r)]
        for j in label1:
            valid_label1[i][j] = 1

        label2 = p.make_all_marrix[(t, r_inv)]
        for j in label2:
            valid_label2[i][j] = 1

        i += 1

        # if (i == 256):
        #     break

    valid_data = torch.LongTensor(valid_data).to(p.device)
    return valid_data, valid_label1, valid_label2


def get_chequer(p):
    ent_perm = np.int32([np.random.permutation(p.dim) for _ in range(p.perm)])
    rel_perm = np.int32([np.random.permutation(p.dim) for _ in range(p.perm)])
    comb_idx = []
    for k in range(p.perm):
        temp = []
        ent_idx, rel_idx = 0, 0

        for i in range(p.matrix_h):
            for j in range(p.matrix_w):
                if k % 2 == 0:
                    if i % 2 == 0:
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                        temp.append(rel_perm[k, rel_idx] + p.dim)
                        rel_idx += 1
                    else:
                        temp.append(rel_perm[k, rel_idx] + p.dim)
                        rel_idx += 1
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                else:
                    if i % 2 == 0:
                        temp.append(rel_perm[k, rel_idx] + p.dim)
                        rel_idx += 1
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                    else:
                        temp.append(ent_perm[k, ent_idx])
                        ent_idx += 1
                        temp.append(rel_perm[k, rel_idx] + p.dim)
                        rel_idx += 1
        comb_idx.append(temp)
    comb_idx = torch.LongTensor(comb_idx).to(p.device)
    return comb_idx
