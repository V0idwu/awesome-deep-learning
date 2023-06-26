import torch
import torch.nn as nn
import torch.nn.functional as F


# Language Mask
class LMPredictHead(nn.Module):
    def __init__(self, config) -> None:
        super(LMPredictHead, self).__init__()
        self.f1 = nn.Linear(config.hidden_dim, config.hidden_dim).to(config.device)
        self.f2 = nn.Linear(config.hidden_dim, config.vocab_size).to(config.device)

    def forward(self, x):
        x = F.relu(self.f1(x))
        return self.f2(x)


# Sentence Classification
class SCLsHead(nn.Module):
    def __init__(self, config) -> None:
        super(SCLsHead, self).__init__()
        self.f1 = nn.Linear(config.hidden_dim, config.hidden_dim).to(config.device)
        self.f2 = nn.Linear(config.hidden_dim, 2).to(config.device)

    def forward(self, x):
        x = F.relu(self.f1(x))
        return self.f2(x)
