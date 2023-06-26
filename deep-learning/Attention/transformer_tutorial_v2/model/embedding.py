import torch
import torch.nn as nn

from .model_config import ModelConfig


class Embedding(nn.Module):
    def __init__(self, config) -> None:
        super(Embedding, self).__init__()
        # id 编码
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_dim).to(config.device)
        self.position_embeddings = nn.Embedding(num_embeddings=config.max_len, embedding_dim=config.hidden_dim).to(
            config.device
        )

        # 预训练时，输入的是两句话
        self.token_type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_dim).to(config.device)
        self.device = config.device

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=self.device)

        we = self.word_embeddings(input_ids)
        pe = self.position_embeddings(position_ids)
        te = self.token_type_embedding(token_type_ids)

        return we + pe + te
