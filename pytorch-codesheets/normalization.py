import torch
import torch.nn as nn


# NOTE: BN 的基本思想是对于每个 mini-batch 中的样本，对其输入的每个特征在 mini-batch 的维度上进行归一化。
#       1. 缓解internal covariate shift问题，提升训练速度
#       2. 缓解梯度爆炸和梯度衰减问题
#       3. 减少初始化参数的影响（从公式来看，初始参数均乘以k倍，网络输出没变，有待商榷）
#       4. 减少过拟合
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # 计算mini-batch的均值和方差
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            # 更新running mean和running var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # 在测试阶段使用running mean和running var
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)
        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        # 重构和缩放
        x = self.gamma.unsqueeze(0) * x + self.beta.unsqueeze(0)
        return x


# NOTE: 与BN不同的是，LN是在特征维度上进行归一化处理。
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # 归一化
        x = (x - mean) / torch.sqrt(var + self.eps)
        # 重构和缩放
        x = self.gamma.unsqueeze(-1) * x + self.beta.unsqueeze(-1)
        return x
