import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class BarcodeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_1 = self.data[idx]
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return sample_1, label

# # own dataset
# X = pd.read_feather("./data/own/X.feather")
# Y = pd.read_feather("./data/own/Y.feather")
#
#
# # deeplexion dataset
# X = pd.read_feather("./data/deeplexion/X.feather")
# Y = pd.read_feather("./data/deeplexion/Y.feather")


# QuipuNet dataset
# X_test = pd.read_feather("./data/quipunet/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy()
# X_dev = pd.read_feather("./data/quipunet/X_dev.feather").to_numpy()
# Y_dev = pd.read_feather("./data/quipunet/Y_dev.feather").to_numpy()
#
# X_dev = torch.from_numpy(X_dev).float()
# X_test = torch.from_numpy(X_test).float()
#
# # reshape
# X_dev = X_dev.reshape((-1, 1, 700))
# X_test = X_test.reshape((-1, 1, 700))
#
# dev_dataset = BarcodeDataset(X_dev, Y_dev)
# test_dataset = BarcodeDataset(X_test, Y_test)
#
# dev_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=False, drop_last=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)


