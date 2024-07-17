import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    SEQ_FEATURES_COL_LEN,
    SEQ_LEN,
    SEQ_TARGETS_COL_LEN,
    SINGLE_FEATURES_COL_LEN,
    SINGLE_TARGETS_COL_LEN,
)


class FFNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super(FFNN, self).__init__()

        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def x_to_seq(x):
    x_seq = x[
        :,
        : SEQ_LEN * SEQ_FEATURES_COL_LEN,
    ].reshape(-1, SEQ_FEATURES_COL_LEN, SEQ_LEN)

    x_flat = (
        x[
            :,
            SEQ_LEN * SEQ_FEATURES_COL_LEN : SEQ_LEN * SEQ_FEATURES_COL_LEN + SINGLE_FEATURES_COL_LEN,
        ]
        .reshape(-1, 1, SINGLE_FEATURES_COL_LEN)
        .repeat(1, SEQ_LEN, 1)
        .transpose(1, 2)
    )

    return torch.cat([x_seq, x_flat], dim=-2)


class CNN(nn.Module):
    def __init__(self, activation=F.relu):
        super(CNN, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(64, 256, 3, padding="same")
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, 3, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, 3, padding="same")
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.bn1(self.activation(self.conv1(x)))
        x = self.bn2(self.activation(self.conv2(x)))
        x = self.bn3(self.activation(self.conv3(x)))
        return x


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.conv1d = nn.Conv1d(25, 64, 1, padding="same")
        self.cnn = CNN()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(64)
        self.conv1d_out = nn.Conv1d(64, 14, 1, padding="same")

    def forward(self, x):
        x = x_to_seq(x)
        global tmp
        tmp = x

        e0 = self.conv1d(x)

        e = self.cnn(e0)
        e = e0 + e + self.global_avg_pool(e)
        e = self.bn(e)
        e = e + self.cnn(e)
        p_all = self.conv1d_out(e)

        p_seq = p_all[:, :SEQ_TARGETS_COL_LEN].flatten(start_dim=-2)
        assert p_seq.shape[-1] == SEQ_LEN * SEQ_TARGETS_COL_LEN

        p_flat = p_all[:, SEQ_TARGETS_COL_LEN:].mean(dim=-1)
        assert p_flat.shape[-1] == SINGLE_TARGETS_COL_LEN

        P = torch.cat([p_seq, p_flat], dim=1)
        return P
