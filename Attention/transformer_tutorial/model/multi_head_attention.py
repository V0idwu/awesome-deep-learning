import math

import torch
import torch.nn as nn

from .model_config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        self.dim = config.hidden_dim
        self.device = config.device

        self.wq = nn.Linear(self.dim, self.dim).to(self.device)
        self.wk = nn.Linear(self.dim, self.dim).to(self.device)
        self.wv = nn.Linear(self.dim, self.dim).to(self.device)

        self.softmax = nn.Softmax(dim=-1)
        self.f = nn.Linear(self.dim, self.dim).to(self.device)
        self.norm = nn.LayerNorm(self.dim).to(self.device)
        self.dropout = nn.Dropout(config.drop_out)

    def split(self, tensor):
        a, b, c = tensor.size()  # a,b,c = batch_size, seq_len, hidden_dim
        d = c // self.n_head  # d = hidden_dim / n_head
        return tensor.view(a, b, self.n_head, d).permute(0, 2, 1, 3)  # [batch_size, seq_len, n_head, d]

    def concat(self, tensor):
        a, b, c, d = tensor.size()
        return tensor.view(a, c, b * d)

    def attention(self, k, q, v):
        _, _, _, d = k.size()
        kt = torch.transpose(k, 2, 3)
        s = (q @ kt) / math.sqrt(d)
        s = self.softmax(s)
        v = s @ v
        return v

    def forward(self, x):  # x: [batch_size, seq_len, hidden_dim]
        k, q, v = self.wk(x), self.wq(x), self.wv(x)  # k, q, v: [batch_size, seq_len, hidden_dim]
        k, q, v = self.split(k), self.split(q), self.split(v)
        v = self.attention(k, q, v)
        v = self.concat(v)

        vb = v

        v = self.norm(v + x)
        v = self.f(v)
        v = self.norm(v + vb)
        return self.dropout(v)
