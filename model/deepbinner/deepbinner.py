import torch
import torch.nn as nn
import torch.nn.functional as F


# class GaussianNoise(nn.Module):
#     def __init__(self, stddev=0.02):
#         super(GaussianNoise, self).__init__()
#         self.stddev = stddev
#
#     def forward(self, x):
#         if self.training:
#             noise = torch.randn_like(x) * self.stddev
#             return x + noise
#         return x


class DeepBinner(nn.Module):
    def __init__(self, class_count):
        super(DeepBinner, self).__init__()

        # self.gaussian_noise = GaussianNoise(stddev=0.02)

        # Initial Conv Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=48, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(48)
        self.dropout1 = nn.Dropout(0.15)

        # Conv Block 2
        self.conv2_1 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.bn2 = nn.BatchNorm1d(48)
        self.dropout2 = nn.Dropout(0.15)

        # Bottleneck Layer
        self.bottleneck = nn.Conv1d(48, 16, kernel_size=1)

        # Conv Block 3
        self.conv3_1 = nn.Conv1d(16, 48, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.bn3 = nn.BatchNorm1d(48)
        self.dropout3 = nn.Dropout(0.15)

        # Conv Block 4
        self.conv4_1 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.bn4 = nn.BatchNorm1d(48)
        self.dropout4 = nn.Dropout(0.15)

        # Inception-style Group
        self.inception_conv1 = nn.Conv1d(48, 48, kernel_size=1, padding=0)
        self.inception_conv2 = nn.Conv1d(48, 48, kernel_size=1, padding=0)
        self.inception_conv3_1 = nn.Conv1d(48, 16, kernel_size=1, padding=0)
        self.inception_conv3_2 = nn.Conv1d(16, 48, kernel_size=3, padding=1)
        self.inception_conv4_1 = nn.Conv1d(48, 16, kernel_size=1, padding=0)
        self.inception_conv4_2 = nn.Conv1d(16, 48, kernel_size=3, padding=1)
        self.inception_conv4_3 = nn.Conv1d(48, 48, kernel_size=3, padding=1)

        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.bn5 = nn.BatchNorm1d(192)  # 48 + 48 + 48 + 48 = 192
        self.dropout5 = nn.Dropout(0.15)

        # Conv Block 5
        self.conv5 = nn.Conv1d(192, 48, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm1d(48)
        self.dropout6 = nn.Dropout(0.15)

        # Conv Block 6
        self.conv6_1 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2)

        self.bn7 = nn.BatchNorm1d(48)
        self.dropout7 = nn.Dropout(0.15)

        # Final Layers
        self.final_conv = nn.Conv1d(48, class_count, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.gaussian_noise(x)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool1(x)

        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.bottleneck(x))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool2(x)

        x = self.bn3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool3(x)

        x = self.bn4(x)
        x = self.dropout4(x)

        # Inception-style group
        x1 = F.relu(self.inception_conv1(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)))
        x2 = F.relu(self.inception_conv2(x))
        x3 = F.relu(self.inception_conv3_1(x))
        x3 = F.relu(self.inception_conv3_2(x3))
        x4 = F.relu(self.inception_conv4_1(x))
        x4 = F.relu(self.inception_conv4_2(x4))
        x4 = F.relu(self.inception_conv4_3(x4))

        # Concatenate the inception outputs
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.pool4(x)

        x = self.bn5(x)
        x = self.dropout5(x)

        x = F.relu(self.conv5(x))
        x = self.bn6(x)
        x = self.dropout6(x)

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = self.pool5(x)

        x = self.bn7(x)
        x = self.dropout7(x)

        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # x = F.softmax(x, dim=1)

        return x

# Usage example:
# model = DeepBinnerNetwork(class_count=NUM_CLASSES)
# output = model(torch.randn(32, 1, input_length))  # Example input
