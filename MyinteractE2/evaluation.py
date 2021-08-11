import torch
import numpy as np


def ranking_and_hits(p, model, test_data_batch):
    # test_data_batch, chequer_perm, perm, matrix_w, matrix_h, ker_size, device

    ranks = []
    ranks_left = []
    ranks_right = []
    hits_one = []
    hits_three = []
    hits_ten = []
    hits_left_one = []
    hits_left_three = []
    hits_left_ten = []
    hits_right_one = []
    hits_right_three = []
    hits_right_ten = []

    for index, data in enumerate(test_data_batch):
        test_data, test_label1, test_label2 = data
        h, r, r_inv, t = test_data[:, 0], test_data[:, 1], test_data[:, 2], test_data[:, 3]
        pred1 = model(p, h, r)
        pred2 = model(p, t, r_inv)

        list1 = torch.arange(pred1.size()[0]).to(p.device)
        list2 = torch.arange(pred2.size()[0]).to(p.device)

        target_pred1 = pred1[list1, t]
        target_pred2 = pred2[list2, h]
        pred1 = torch.where(test_label1.byte(), torch.zeros_like(pred1), pred1)
        pred2 = torch.where(test_label2.byte(), torch.zeros_like(pred2), pred2)
        pred1[list1, t] = target_pred1
        pred2[list2, h] = target_pred2
        ranks_list1 = torch.argsort(torch.argsort(pred1, dim=1, descending=True), dim=1, descending=False) + 1
        ranks_list2 = torch.argsort(torch.argsort(pred2, dim=1, descending=True), dim=1, descending=False) + 1

        rank1_nplist = ranks_list1[list1, t].cpu().numpy()
        rank2_nplist = ranks_list2[list2, h].cpu().numpy()
        # @1
        one_left = np.where(rank1_nplist <= 1, 1.0, 0.0).tolist()
        one_right = np.where(rank2_nplist <= 1, 1.0, 0.0).tolist()
        hits_one.extend(one_left + one_right)
        hits_left_one.extend(one_left)
        hits_right_one.extend(one_right)

        # @3
        three_left = np.where(rank1_nplist <= 3, 1.0, 0.0).tolist()
        three_right = np.where(rank2_nplist <= 3, 1.0, 0.0).tolist()
        hits_three.extend(three_left + three_right)
        hits_left_three.extend(three_left)
        hits_right_three.extend(three_right)

        # @10
        ten_left = np.where(rank1_nplist <= 10, 1.0, 0.0).tolist()
        ten_right = np.where(rank2_nplist <= 10, 1.0, 0.0).tolist()
        hits_ten.extend(ten_left + ten_right)
        hits_left_ten.extend(ten_left)
        hits_right_ten.extend(ten_right)

        # 添加rank
        rank1_list = rank1_nplist.tolist()
        rank2_list = rank2_nplist.tolist()
        ranks.extend(rank1_list + rank2_list)
        ranks_left.extend(rank1_list)
        ranks_right.extend(rank2_list)

    print("@1 总命中率：{}".format(np.mean(hits_one)))
    print("@1 left命中率：{}".format(np.mean(hits_left_one)))
    print("@1 right命中率：{}".format(np.mean(hits_right_one)))

    print("@3 总命中率：{}".format(np.mean(hits_three)))
    print("@3 left命中率：{}".format(np.mean(hits_left_three)))
    print("@3 right命中率：{}".format(np.mean(hits_right_three)))

    print("@10 总命中率：{}".format(np.mean(hits_ten)))
    print("@10 left命中率：{}".format(np.mean(hits_left_ten)))
    print("@10 right命中率：{}".format(np.mean(hits_right_ten)))

    print("MR_all:{}".format(np.mean(ranks)))
    print("MR_left:{}".format(np.mean(ranks_left)))
    print("MR_right:{}".format(np.mean(ranks_right)))

    print("MRR_all:{}".format(np.mean(1.0 / np.array(ranks))))
    print("MRR_left:{}".format(np.mean(1.0 / np.array(ranks_left))))
    print("MRR_right:{}".format(np.mean(1.0 / np.array(ranks_right))))
