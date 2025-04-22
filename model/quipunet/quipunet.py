import torch.nn as nn


hp = {
    "traceLength": 700,
    "traceTrim": 0,
    "barcodes": 8,  # distinct barcode count
    "normalise_levels": True,  # wherther to normalise experiments per batch before feetingh into NN
}

class QuipuModel(nn.Module):
    def __init__(self, num_class):
        super(QuipuModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),  # Batch normalization for Conv1D has only 1 dimension
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.25)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.25)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.25)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(256 * (700 // 27), 512),  # Adjust output size according to pooling
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc3 = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x= self.fc3(x)
        return x

