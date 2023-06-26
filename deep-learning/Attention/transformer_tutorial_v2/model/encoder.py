import torch
import torch.nn as nn

from .model_config import ModelConfig
from .multi_head_attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(config)

    def forward(self, x):
        return self.attention(x)
