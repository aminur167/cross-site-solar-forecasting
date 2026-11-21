"""
PyTorch Deep Learning Sequence Architectures for Solar PV Power Forecasting
Implements Recurrent Neural Networks (LSTM) and Temporal Convolutional Networks (TCN).
Designed for sequence modeling with sliding lookback windows (T=12 hours).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TimeSeriesSequenceDataset(Dataset):
    """
    Sliding window dataset for sequence models.
    At time t, takes features [t-seq_len+1 : t] to predict y(t+1).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 6, timestamps=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        if timestamps is None:
            self.end_indices = np.arange(seq_len - 1, len(self.X))
        else:
            ts = pd.DatetimeIndex(timestamps)
            if len(ts) != len(self.X):
                raise ValueError("timestamps must align with X and y")
            self.end_indices = np.array([
                end for end in range(seq_len - 1, len(ts))
                if np.all(ts[end - seq_len + 1:end + 1].to_series().diff().iloc[1:] == pd.Timedelta(hours=1))
            ], dtype=int)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx):
        end = int(self.end_indices[idx])
        x_seq = self.X[end - self.seq_len + 1 : end + 1]
        y_target = self.y[end]
        return x_seq, y_target

class SolarLSTM(nn.Module):
    """
    Two-layer LSTM with dropout and linear readout.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :]
        preds = self.fc(last_out).squeeze(-1)
        return torch.clamp(preds, min=0.0, max=1.2)

class Chomp1d(nn.Module):
    """Ensures causality by trimming padding from the future."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Dilated Causal 1D Convolutional Residual Block for TCN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SolarTCN(nn.Module):
    """
    Temporal Convolutional Network with causal dilations [1, 2, 4].
    """
    def __init__(self, input_dim: int, num_channels: list = [32, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x input: [batch, seq_len, input_dim] -> conv1d needs [batch, input_dim, seq_len]
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        last_out = y[:, :, -1]
        preds = self.fc(last_out).squeeze(-1)
        return torch.clamp(preds, min=0.0, max=1.2)
