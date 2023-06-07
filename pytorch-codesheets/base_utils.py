#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@Time    :   2023/06/07 16:52:41
@Author  :   Tianyi Wu 
@Contact :   wutianyitower@hotmail.com
@File    :   base_utils.py
@Version :   1.0
@Desc    :   None
"""

import random

import numpy as np
import torch


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#   NOTE: 对于不同线程的随机数种子设置，
#         主要通过DataLoader的worker_init_fn参数来实现。
#         默认情况下使用线程ID作为随机数种子。
#         如果需要自己设定，可以参考以下代码：
#         dataloader = DataLoader(dataset, batch_size=4, shuffle=True, worker_init_fn=worker_init_fn)
#         GLOBAL_SEED = 1
#         GLOBAL_WORKER_ID = None
#         def worker_init_fn(worker_id):
#             global GLOBAL_WORKER_ID
#             GLOBAL_WORKER_ID = worker_id
#             setup_seed(GLOBAL_SEED + worker_id)


def setup_cuda(cuda: bool = True, torch_deterministic: bool = False, verbose: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")

    if verbose:
        print("--------------- Device Info --------------")
        print(f"{'Current device':<16}: {device}")
        print(f"{'Pytorch version':<16}: {torch.__version__}")
        print(f"{'CUDA version':<16}: {torch.version.cuda}")
        print(f"{'cuDNN version':<16}: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available() and cuda:
            print(f"{'Device count':<16}: {torch.cuda.device_count()}")
            print(f"{'Device name':<16}: {torch.cuda.get_device_name(device)}")
        print("------------------------------------------\n")

    torch.backends.cudnn.deterministic = torch_deterministic  # significantly slowed down
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float32)

    return device
