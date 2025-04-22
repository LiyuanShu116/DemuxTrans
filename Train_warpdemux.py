import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
import joblib
import time
from Test import test, dev
from dataset import BarcodeDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from dataset import BarcodeDataset
from torch.utils.tensorboard import SummaryWriter
from utils.Quipu import augment
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from data.deeplexion.Signal2Image import rain.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/quipunet/X_tesignal2image
# from model.quipunet.quipunet import QuipuModel
# from model.model.multi_scale import Model_TCN
# from model.deepbinner.deepbinner import DeepBinner
# from model.WarpDemuX.WarpDemuX_main.warpdemux.models.dtw_svm import DTW_SVM_Model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
#
# # X_train = pd.read_feather("./data/quipunet/X_train.feather").to_numpy()
# # Y_train = pd.read_feather("./data/quipunet/Y_tst.feather").to_numpy()
# Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("./data/quipunet/X_dev.feather").to_numpy()
# Y_dev = pd.read_feather("./data/quipunet/Y_dev.feather").to_numpy().squeeze()

# # own data
X_train = pd.read_feather("./data/own/X_train.feather").to_numpy()
Y_train = pd.read_feather("./data/own/Y_train.feather").to_numpy().squeeze()
X_test = pd.read_feather("./data/own/X_test.feather").to_numpy()
Y_test = pd.read_feather("./data/own/Y_test.feather").to_numpy().squeeze()
X_dev = pd.read_feather("./data/own/X_val.feather").to_numpy()
Y_dev = pd.read_feather("./data/own/Y_val.feather").to_numpy().squeeze()

model = DTW_SVM_Model()
# model.fit(X_train, Y_train, block_size=3000)

# save_path = 'model_saved/warpdemux_dataset1_0.7.pkl'
save_path = 'model_saved/warpdemux_dataset3_5.pkl'
# joblib.dump(model, save_path)

model = joblib.load(save_path)

start_time = time.time()
y_probs = model.predict(X_test, return_df=False)
y_preds = y_probs[0]
y_preds[y_preds == -1] = 3

accuracy = accuracy_score(Y_test, y_preds)
precision = precision_score(Y_test, y_preds, average='weighted')
recall = recall_score(Y_test, y_preds, average='weighted')
f1 = f1_score(Y_test, y_preds, average='weighted')
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)



