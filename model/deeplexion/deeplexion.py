import torch.nn as nn
import torch.nn.functional as F

# from Train import deeplexion_train
from Test import test, dev
from dataset import BarcodeDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.deeplexion.Signal2Image import signal2image


class ResNetLayer(nn.Module):
    def __init__(self, in_channel, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
        super(ResNetLayer, self).__init__()

        self.batch_normalization = nn.BatchNorm2d(in_channel) if batch_normalization else None
        self.batch_normalization1 = nn.BatchNorm2d(num_filters) if batch_normalization else None

        self.activation = nn.ReLU() if activation == 'relu' else None
        self.conv_first = conv_first

        if self.conv_first:
            self.conv = nn.Conv2d(in_channels=1,  # Assuming input channels are 3 (e.g., RGB image)
                                  out_channels=num_filters,
                                  kernel_size=kernel_size,
                                  stride=strides,
                                  padding=(kernel_size // 2),  # 'same' padding
                                  bias=not batch_normalization)  # No bias if batch normalization is used

        else:
            self.conv = nn.Conv2d(in_channels=in_channel,  # Assuming input channels are 3 (e.g., RGB image)
                                  out_channels=num_filters,
                                  kernel_size=kernel_size,
                                  stride=strides,
                                  padding=(kernel_size // 2),  # 'same' padding
                                  bias=not batch_normalization)  # No bias if batch normalization is used

    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = self.batch_normalization1(x)
            if self.activation:
                x = self.activation(x)
        else:
            if self.batch_normalization:
                x = self.batch_normalization(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)

        return x

class DeeplexionNet(nn.Module):
    def __init__(self, depth=20, num_classes=4):
        super(DeeplexionNet, self).__init__()

        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

        self.num_filters_in = 16
        self.num_res_blocks = (depth - 2) // 9      # 2

        # Initial convolution layer
        self.conv1 = ResNetLayer(in_channel=1, num_filters=self.num_filters_in, conv_first=True)

        # Stages
        self.stages = nn.ModuleList()

        in_channel = [16, 64, 64, 128, 128, 256]
        i = 0

        for stage in range(3):
            stage_layers = []
            for res_block in range(self.num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = self.num_filters_in * 4     # 64
                    # self.in_channel = 16
                    if res_block == 0:      # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = self.num_filters_in * 2  # 32
                    if res_block == 0:
                        strides = 2  # downsample


                # Bottleneck residual unit
                y = ResNetLayer(in_channel=in_channel[i], num_filters=self.num_filters_in, kernel_size=1, strides=strides,
                                activation=activation, batch_normalization=batch_normalization, conv_first=False)
                stage_layers.append(y)

                y = ResNetLayer(in_channel=self.num_filters_in, num_filters=self.num_filters_in, conv_first=False)
                stage_layers.append(y)

                y = ResNetLayer(in_channel=self.num_filters_in, num_filters=num_filters_out, kernel_size=1, conv_first=False)
                stage_layers.append(y)

                i = i + 1

            self.stages.append(nn.Sequential(*stage_layers))
            self.num_filters_in = num_filters_out      # 64 128 256

        self.shortcut = []
        self.shortcut1 = ResNetLayer(in_channel=16, num_filters=64, kernel_size=1, strides=1,
                                     activation=None, batch_normalization=False, conv_first=False)
        self.shortcut.append(self.shortcut1)
        self.shortcut2 = ResNetLayer(in_channel=64, num_filters=128, kernel_size=1, strides=2,
                                     activation=None, batch_normalization=False, conv_first=False)
        self.shortcut.append(self.shortcut2)
        self.shortcut3 = ResNetLayer(in_channel=128, num_filters=256, kernel_size=1, strides=2,
                                     activation=None, batch_normalization=False, conv_first=False)
        self.shortcut.append(self.shortcut3)
        # Classifier
        self.batch_norm = nn.BatchNorm2d(self.num_filters_in)
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(12544, num_classes)

    def forward(self, x):
        x = self.conv1(x)                        #(batch, 16, 224, 224)

        # for stage in self.stages:
        for i in range(3):
            stage = self.stages[i]
            residual = x
            x = stage[0](x)                       #(batch, 16, 224, 224)
            x = stage[1](x)                       #(batch, 16, 224, 224)
            x = stage[2](x)                       #(batch, 64, 224, 224)
            x = x + self.shortcut[i](residual)
            x = stage[3](x)
            x = stage[4](x)
            x = stage[5](x)                       #(batch, 64, 224, 224)


        # Classifier
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pool(x)  # Global average pooling
        batch = x.size(0)
        # x = torch.flatten(x, 1)  # Flatten the output
        x = x.view(batch, -1)
        x = self.fc(x)  # Fully connected layer
        return x

# model = DeeplexionNet(num_classes=4)
#
# # own data
# X_train = pd.read_feather("../../data/own/X_train.feather").to_numpy()
# Y_train = pd.read_feather("../../data/own/Y_train.feather").to_numpy().squeeze()
# X_test = pd.read_feather("../../data/own/X_test.feather").to_numpy()
# Y_test = pd.read_feather("../../data/own/Y_test.feather").to_numpy().squeeze()
# X_dev = pd.read_feather("../../data/own/X_val.feather").to_numpy()
# Y_dev = pd.read_feather("../../data/own/Y_val.feather").to_numpy().squeeze()
#
# # deeplexion data
# # X_train = pd.read_feather("../../data/deeplexion/X_train.feather").to_numpy()
# # Y_train = pd.read_feather("../../data/deeplexion/Y_train.feather").to_numpy().squeeze()
# # X_test = pd.read_feather("../../data/deeplexion/X_test.feather").to_numpy()
# # Y_test = pd.read_feather("../../data/deeplexion/Y_test.feather").to_numpy().squeeze()
# # X_dev = pd.read_feather("../../data/deeplexion/X_val.feather").to_numpy()
# # Y_dev = pd.read_feather("../../data/deeplexion/Y_val.feather").to_numpy().squeeze()
#
# # quipunet data
# # X_train = pd.read_feather("../../data/quipunet/X_train.feather").to_numpy()
# # Y_train = pd.read_feather("../../data/quipunet/Y_train.feather").to_numpy().squeeze()
# # X_test = pd.read_feather("../../data/quipunet/X_test.feather").to_numpy()
# # Y_test = pd.read_feather("../../data/quipunet/Y_test.feather").to_numpy().squeeze()
# # X_dev = pd.read_feather("../../data/quipunet/X_dev.feather").to_numpy()
# # Y_dev = pd.read_feather("../../data/quipunet/Y_dev.feather").to_numpy().squeeze()
#
# # X_dev = torch.from_numpy(X_dev).float()
# # X_test = torch.from_numpy(X_test).float()
# #
# # dev_dataset = BarcodeDataset(X_dev, Y_dev)
# # test_dataset = BarcodeDataset(X_test, Y_test)
# #
# # dev_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=False, drop_last=False)
# # test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
#
# X_train = signal2image(X_train)
# X_test = signal2image(X_test)
# X_dev = signal2image(X_dev)
#
#
# train_dataset = BarcodeDataset(X_train, Y_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_dataset = BarcodeDataset(X_test, Y_test)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
# dev_dataset = BarcodeDataset(X_dev, Y_dev)
# dev_loader = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=False, drop_last=False)
#
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# model = DeeplexionNet()
# model.to(device)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
# writer = SummaryWriter('../logs_train/deeplexion')
#
# num_epochs = 50
# for epoch in range(num_epochs):
#     deeplexion_train(model, train_loader, optimizer, epoch, device, criterion, scheduler, writer)
#     print(f'Epoch [{epoch + 1}/{num_epochs}]')
#     deeplexion_test(epoch, model, test_loader, device, writer)
#     deeplexion_dev(epoch, model, dev_loader, device, writer)
#     # test(epoch, model, test_loader, device, writer)
#     # dev(epoch, model, dev_loader, device, writer)
#
# print('Finished Training')
# torch.save(model.state_dict(), '../../model_saved/deeplexion_dataset3.pth')
# writer.close()
# #
# #
#
#
