import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os

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
from data.deeplexion.Signal2Image import signal2image

from model.quipunet.quipunet import QuipuModel
from model.model.multi_scale import Model_TCN
from model.deepbinner.deepbinner import DeepBinner
from model.deeplexion.deeplexion import DeeplexionNet
# from model.WarpDemuX.WarpDemuX import DTW_SVM_Model

def train(model, X_train, Y_train, optimizer, epoch, device, criterion, scheduler, writer):
    model.train()
    running_loss = 0.0

    X = np.repeat(X_train, 1, axis=0)  # make copies
    X = augment.magnitude(X, std=0.08)
    X = augment.stretchDuration(X, std=0.1, probability=0.3)
    X = augment.addNoise(X, std=0.08)

    X = torch.from_numpy(X).float()
    X = X.reshape((-1, 1, 700))
    train_dataset = BarcodeDataset(X, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        running_loss += loss.item()

        if i % 10 == 9:
            writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    scheduler.step()

def deeplexion_train(model, train_loader, optimizer, epoch, device, criterion, scheduler, writer):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        running_loss += loss.item()

        if i % 10 == 9:
            writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    scheduler.step()



# own data
# X_train = pd.read_feather("./data/own/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/own/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/own/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/own/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("./data/own/X_val.feather").to_numpy()
# Y_dev = pd.read_feather("./data/own/Y_val.feather").to_numpy().squeeze()

# deeplexion data
# X_train = pd.read_feather("./data/deeplexion/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/deeplexion/Y_train.feather").to_numpy().squeeze()
X_test = pd.read_feather("./data/deeplexion/X_test.feather").to_numpy()
Y_test = pd.read_feather("./data/deeplexion/Y_test.feather").to_numpy().squeeze()
X_dev = pd.read_feather("./data/deeplexion/X_val.feather").to_numpy()
Y_dev = pd.read_feather("./data/deeplexion/Y_val.feather").to_numpy().squeeze()

# quipunet data
# X_train = pd.read_feather("./data/quipunet/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/quipunet/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/quipunet/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("./data/quipunet/X_dev.feather").to_numpy()
# Y_dev = pd.read_feather("./data/quipunet/Y_dev.feather").to_numpy().squeeze()

# model = Model_TCN(
#         num_layers=3,   #比对
#         d_model=256,
#         num_heads=4,
#         d_ff=1024,
#         num_classes=8,
#         #hidden_dim=128,
#         dropout=0.1,
#         num_channel=48
# )
# model.load_state_dict(torch.load('./model_saved/data1_all.pth'))
#
# model = Model_TCN(
#         num_layers=3,   #比对
#         d_model=256,
#         num_heads=8,
#         d_ff=1024,
#         num_classes=4,
#         #hidden_dim=128,
#         dropout=0.1,
#         num_channel=64
# )
# model.load_state_dict(torch.load('./model_saved/data3_head_8.pth'))
#
# model.fc = nn.Sequential(
#     nn.Linear(in_features=87*64, out_features=1024),
#     nn.ELU(),
#     nn.Linear(in_features=1024, out_features=256),
#     nn.ELU(),
#     nn.Linear(in_features=256, out_features=8)  # Update output classes
# )

# model = QuipuModel(num_class=4)
# model.load_state_dict(torch.load('./model_saved/quipunet_dataset3.pth'))
# model.fc3 = nn.Linear(in_features=512, out_features=8)
# model_QuipuNet.to(device)

#
# model = DeepBinner(class_count=4)
# model_DeepBinner.load_state_dict(torch.load('./model_saved/deepbinner_dataset1.pth'))
# model_DeepBinner.to(device)

model = DeeplexionNet(num_classes=4)
model.load_state_dict(torch.load('./model_saved/deeplexion_dataset3.pth'))
model.fc = nn.Linear(12544, 8)
# model_DeeplexionNet.to(device)

for param in model.parameters():
    param.requires_grad = True


