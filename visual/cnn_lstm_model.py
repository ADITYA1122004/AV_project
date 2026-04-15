import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(128 * 12 * 12, 128)

    def forward(self, x):
        # x: (batch, 1, 96, 96)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LipReadingModel(nn.Module):
    def __init__(self, hidden_size=128, num_classes=30):
        super(LipReadingModel, self).__init__()

        self.cnn = CNNEncoder()

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1, 96, 96)

        batch_size, seq_len, C, H, W = x.size()

        cnn_features = []

        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.cnn(frame)
            cnn_features.append(feat)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, _ = self.lstm(cnn_features)

        out = self.fc(lstm_out[:, -1, :])

        return out

