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

import logging
import os
import random
import time

import numpy as np
import torch


# NOTE: 为了保证实验的可重复性，需要设置随机数种子。
def setup_seed(seed: int = 1, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic  # CUDA运算的确定性
    torch.backends.cudnn.benchmark = not torch_deterministic  # 数据变化的情况下，减少网络效率的变化


# NOTE: 对于不同线程的随机数种子设置，主要通过DataLoader的worker_init_fn参数来实现。默认情况下使用线程ID作为随机数种子
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, worker_init_fn=worker_init_fn)
def worker_init_fn(worker_id, global_seed=1):
    setup_seed(global_seed + worker_id)


# NOTE: CPU和GPU的设备设置
def setup_device(cuda: bool = True, device_ids: list = ["0"], verbose: bool = False):
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available() and cuda:
        assert torch.cuda.device_count() > 0, "No GPU found, please run without --cuda"
        assert torch.cuda.device_count() >= len(
            device_ids
        ), f"Choosen GPU number {len(device_ids)} is larger than available GPU number {torch.cuda.device_count()}"

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
        if len(device_ids) == 1:
            info = {"device": torch.device(f"cuda:{device_ids[0]}"), "device_ids": ["0"]}
        elif len(device_ids) > 1:
            info = {"device": None, "device_ids": device_ids}
    else:
        info = {"device": torch.device("cpu"), "device_ids": []}

    if verbose:
        print("---------------- Lib Info ----------------")
        print(f"{'Pytorch version':<16}: {torch.__version__}")
        print(f"{'CUDA version':<16}: {torch.version.cuda}")
        print(f"{'cuDNN version':<16}: {torch.backends.cudnn.version()}")

        print("--------------- Device Info --------------")
        if torch.cuda.is_available() and cuda and len(device_ids) > 0:
            for device_id in device_ids:
                tmp = torch.device(f"cuda:{device_id}")
                print(f"{'Choosen device':<16}: {tmp}")
                print(f"{'Device name':<16}: {torch.cuda.get_device_name(tmp)}")
        else:
            print(f"{'Choosen device':<16}: {info['device']}")
        print("------------------------------------------\n")

    return info


# NOTE:计算pytorch cuda运行时间，由于pytorch在cuda上的计算都是异步的，
# 如果直接使用time.time()来记录时间差并不能计算得到真实的时间，需要调用
# torch.cuda.synchronize()来同步时间。
def calculate_time(func):
    torch.cuda.synchronize()
    t0 = time.time()
    func()
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0


# NOTE: Print log to console
class ConsoleLogger:
    def __init__(self, log_name: str = "default", log_level: int = logging.DEBUG) -> None:
        self.__name = log_name
        self.__level = log_level
        self.init_stream_handler()

    def init_stream_handler(self) -> None:
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.__level)
        self.stream_handler.setFormatter(
            logging.Formatter(
                fmt=f"[%(asctime)s] [{self.__name}] [%(filename)s %(threadName)s -> %(funcName)s line:%(lineno)d] [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def get_logger(self) -> logging.Logger:
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(self.__level)
        self.__logger.addHandler(self.stream_handler)
        return self.__logger

    @property
    def level(self):
        return self.__level

    @property
    def name(self):
        return self.__name
