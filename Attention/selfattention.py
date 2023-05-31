import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(SelfAttention, self).__init__()
        self.scale = dk**-0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x


att = SelfAttention(dim=2, dk=2, dv=3)
x = torch.rand((1, 4, 2))
output = att(x)

print(output)
