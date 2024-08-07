import torch
import torch.nn as nn

from .embedding import Embedding
from .model_config import ModelConfig
from .multi_head_attention import MultiHeadAttention
from .pooler import Pooler


# Encoder
class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super(Transformer, self).__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([MultiHeadAttention(config) for _ in range(config.layer_count)])

        self.pooler = Pooler(config)
        self.device = config.device

    def forward(self, input_ids, token_type_ids=None):
        em = self.embedding(input_ids, token_type_ids)  # [batch_size, seq_len, hidden_dim]
        for layer in self.layers:
            em = layer(em)
        p = self.pooler(em)
        return p, em
