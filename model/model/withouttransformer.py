import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from torch.nn.utils import weight_norm

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, depth)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.split_heads(self.wq(x), batch_size)  # (batch_size, num_heads, seq_length, depth)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        # Scaled dot-product attention
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)  # (batch_size, seq_length, d_model)
        return self.dense(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        #self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #return self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(x))))
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MHA(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.mha(x)
        x = self.layernorm1(x + self.dropout1(attn_output))  # Residual connection
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output))  # Residual connection


class Embed(nn.Module):
    def __init__(self, block_size=1, embedding_size=128):     #block_size=10
        super(Embed, self).__init__()
        self.block_size = block_size
        self.embedding_size = embedding_size

        #self.conv1 = nn.Conv2d(1, 128, (1, block_size), stride=(1, block_size))
        self.conv1 = nn.Conv1d(in_channels=block_size, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=self.embedding_size, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(self.embedding_size)
        self.pool1 = nn.AvgPool1d(2)
        self.pool2 = nn.AvgPool1d(2)
        self.linear = nn.Linear(in_features=256, out_features=256)

        self.downsample1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4)
        )
        self.downsample2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2)
        )


    def forward(self, data):    #(n, 1, 700)
        # data = data.view(data.size(0), -1, self.block_size
        identity1 = self.downsample1(data)  # (n, 32, 175)

        x = self.conv1(data)        # Shape: (n, 32, 700)
        x = self.batch_norm1(x)
        x = self.pool1(F.elu(x))    # (n, 32, 350)
        identity2 = self.downsample2(x)     # (n, 32, 175)


        x = self.conv2(x)  # Shape: (n, 64, 350)
        x = self.batch_norm2(x)
        x = self.pool2(F.elu(x))  # (n, 64, 175)
        identity3 = x

        x = self.conv3(x)  # Shape: (n, 128, 175)
        x = self.batch_norm3(x)
        x = F.elu(x)       # (n, 128, 175)

        x = torch.cat((identity1, identity2, identity3, x), dim=1)  #(n, 256, 175)

        x = x.permute(0, 2, 1)  # Shape: (n, 175, 256)

        # x = F.elu(self.linear(x))
        # x = F.elu(self.linear(x))

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model_TCN(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, num_classes, dropout=0.1, num_channel=16):
        super(Model_TCN, self).__init__()
        self.embed = Embed(block_size=1, embedding_size=128)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        #
        # self.enc_layers1 = nn.ModuleList([
        #     TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)
        # ])

        self.tcn = TemporalConvNet(num_inputs=256, num_channels=[512, 256, 128, num_channel], kernel_size=3, dropout=0.2)
        self.pool = nn.AvgPool1d(2)


        # flatten
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_channel*87, out_features=1024),
            nn.ELU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):         # (n, 1, 700)
        out2 = self.embed(x)           # (n, 175, 480)

        # out2 = self.pos_encoding(out2)
        # for layer in self.enc_layers1:    # time-step
        #     out2 = layer(out2)

        out2 = out2.permute(0, 2, 1)             # (batch, 480, 175)
        out2 = self.tcn(out2)                    # (batch, 64, 175)
        out2 = self.pool(out2)                   # (batch, 64, 87)

        batch = out2.size(0)
        out = out2.reshape(batch, -1)    #(batch, 64*87)
        # feature = out
        x = self.fc(out)

        return x




























