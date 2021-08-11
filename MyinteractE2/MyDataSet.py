import torch.utils.data as Data


class My_Train_DataSet(Data.Dataset):
    def __init__(self, train_data, train_label):
        super(My_Train_DataSet, self).__init__()
        self.train_data = train_data
        self.train_label = train_label

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_label[idx]


class My_Test_DataSet(Data.Dataset):
    def __init__(self, test_data, test_label1, test_label2):
        super(My_Test_DataSet, self).__init__()
        self.test_data = test_data
        self.test_label1 = test_label1
        self.test_label2 = test_label2

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        return self.test_data[idx], self.test_label1[idx], self.test_label2[idx]
