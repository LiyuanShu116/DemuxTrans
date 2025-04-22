import joblib
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
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
from data.deeplexion.Signal2Image import signal2image

from model.quipunet.quipunet import QuipuModel
from model.model.multi_scale import Model_TCN
from model.deepbinner.deepbinner import DeepBinner
from model.WarpDemuX.WarpDemuX_main.warpdemux.models.dtw_svm import DTW_SVM_Model
from model.deeplexion.deeplexion import DeeplexionNet


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

# def deeplexion_train(model, train_loader, optimizer, epoch, device, criterion, scheduler, writer):
#     model.train()
#     running_loss = 0.0
#
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs = inputs.to(device, dtype=torch.float32)
#         labels = labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # scheduler.step()
#         running_loss += loss.item()
#
#         if i % 10 == 9:
#             writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
#             running_loss = 0.0
#
#     scheduler.step()
#
# # own data
# X_train = pd.read_feather("./data/own/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/own/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/own/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/own/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("./data/own/X_val.feather").to_numpy()
# Y_dev = pd.read_feather("./data/own/Y_val.feather").to_numpy().squeeze()
#
# # deeplexion data
# X_train = pd.read_feather("./data/deeplexion/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/deeplexion/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/deeplexion/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/deeplexion/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("./data/deeplexion/X_val.feather").to_numpy()
# Y_dev = pd.read_feather("./data/deeplexion/Y_val.feather").to_numpy().squeeze()
#
# # quipunet data
X_train = pd.read_feather("./data/quipunet/X_train.feather").to_numpy()
Y_train = pd.read_feather("./data/quipunet/Y_train.feather").to_numpy().squeeze()
X_test = pd.read_feather("./data/quipunet/X_test.feather").to_numpy()
Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()
X_dev = pd.read_feather("./data/quipunet/X_dev.feather").to_numpy()
Y_dev = pd.read_feather("./data/quipunet/Y_dev.feather").to_numpy().squeeze()
#
# HyxDemux data
# X_train = pd.read_feather("./data/4/D3/X_train.feather").to_numpy()
# Y_train = pd.read_feather("./data/4/D3/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("./data/4/D3/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/4/D3/Y_test.feather").to_numpy().squeeze()
# X = pd.read_feather("./data/4/D1/X.feather").to_numpy()
# Y = pd.read_feather("./data/4/D1/Y.feather").to_numpy().squeeze()



X_dev = torch.from_numpy(X_dev).float()
X_test = torch.from_numpy(X_test).float()
# X = torch.from_numpy(X).float()
# X_test = signal2image(X_test)

#
# # reshape
X_dev = X_dev.reshape((-1, 1, 700))
X_test = X_test.reshape((-1, 1, 700))
# X = X.reshape((-1, 1, 700))
#
dev_dataset = BarcodeDataset(X_dev, Y_dev)
test_dataset = BarcodeDataset(X_test, Y_test)
# dataset = BarcodeDataset(X, Y)
#
dev_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
# loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, drop_last=False)
#
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#
# tcn model
model = Model_TCN(
        num_layers=3,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        num_classes=8,
        #hidden_dim=128,
        dropout=0.1,
        num_channel=48
)
# model.load_state_dict(torch.load('./model_saved/data1_all.pth'))

# model = Model_TCN(
#         num_layers=3,
#         d_model=256,
#         num_heads=8,
#         d_ff=1024,
#         num_classes=4,
#         #hidden_dim=128,
#         dropout=0.1,
#         num_channel=64
# )
# model.load_state_dict(torch.load('./model_saved/data3_head_8.pth'))

# model = DeepBinner(class_count=8)
# model.load_state_dict(torch.load('./model_saved/deepbinner_dataset1.pth'))

# model = QuipuModel(num_class=8)
# model.load_state_dict(torch.load('./model_saved/quipunet_dataset1.pth'))

# model= DeeplexionNet(num_classes=4)
# model.load_state_dict(torch.load('./model_saved/deeplexion_dataset3.pth'))

# warpdemux
# model = DTW_SVM_Model()
# model.fit(X_train, Y_train, block_size=2000)
# save_path = 'model_saved/warpdemux_dataset1.pkl'
# save_path = 'model_saved/warpdemux_dataset3.pkl'
# joblib.dump(model, save_path)
#
#
model.to(device)
#
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
writer = SummaryWriter('../logs_train')

#
best_acc = 0.0
save_path = 'model_saved/1.pth'

start_time = time.time()
best_acc = test(1, model, test_loader, device, save_path, best_acc)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)

num_epochs = 200
for epoch in range(num_epochs):
    train(model, X_train, Y_train, optimizer, epoch, device, criterion, scheduler, writer)
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    best_acc = test(epoch, model, test_loader, device, save_path, best_acc)
    dev(epoch, model, dev_loader, device)

# print(f"Training completed. Best model saved with accuracy: {100 * best_acc:.4f}%")

print('Finished Training')

# # tcn
# # torch.save(model.state_dict(), './model_saved/model_dataset3.pth')
# # # cnn
# # torch.save(model.state_dict(), './model_saved/cnn.pth')
# # # cnn_lstm
# # torch.save(model.state_dict(), './model_saved/cnn_lstm.pth')
# # # transformer
# # torch.save(model.state_dict(), './model_saved/transformer.pth')
# # # quipunet
# # torch.save(model.state_dict(), './model_saved/quipunet.pth')
# writer.close()