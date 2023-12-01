#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@Time    :   2023/11/28 15:00:10
@Author  :   Tianyi Wu 
@Contact :   wutianyitower@hotmail.com
@File    :   dl_utils.py
@Version :   1.0
@Desc    :   None
"""


import logging
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib_inline import backend_inline

dl_utils = sys.modules[__name__]


# NOTE: 为了保证实验的可重复性，需要设置随机数种子。
def setup_config(seed: int = 227):
    """set random seed for pytorch, numpy and random library."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # NOTE 注意事项1：适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    #      反之，如果卷积层的设置一直变化，网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    torch.backends.cudnn.benchmark = True

    # NOTE 注意事项2: Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动，
    #      设置：
    torch.backends.cudnn.deterministic = True

    # 设置为True，说明设置为使用使用非确定性算法
    torch.backends.cudnn.enabled = True

    """set default data type for pytorch"""
    torch.set_default_dtype(torch.float32)
    # torch.set_default_tensor_type(torch.FloatTensor)


# NOTE: 对于不同线程的随机数种子设置，主要通过DataLoader的worker_init_fn参数来实现。默认情况下使用线程ID作为随机数种子
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, worker_init_fn=worker_init_fn)
def worker_init_fn(worker_id, global_seed=1):
    setup_config(global_seed + worker_id)


def try_gpu(i: int = 0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


# NOTE: CPU和GPU的设备设置
def try_device(cuda: bool = True, device_ids: list = ["0"], verbose: bool = False):
    if torch.cuda.is_available() and cuda:
        assert torch.cuda.device_count() > 0, "No GPU found, please run without --cuda"
        assert torch.cuda.device_count() >= len(
            device_ids
        ), f"Choosen GPU number {len(device_ids)} is larger than available GPU number {torch.cuda.device_count()}"

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
        if len(device_ids) == 1:
            devices = [torch.device(f"cuda:{device_ids[0]}")]
        elif len(device_ids) > 1:
            devices = [torch.device(f"cuda:{i}") for i in devices]
    else:
        devices = [torch.device("cpu")]

    if verbose:
        print("---------------- Lib Info ----------------")
        print(f"{'Pytorch version':<16}: {torch.__version__}")
        print(f"{'CUDA version':<16}: {torch.version.cuda}")
        print(f"{'cuDNN version':<16}: {torch.backends.cudnn.version()}")

        print("--------------- Device Info --------------")
        if torch.cuda.is_available() and cuda and len(devices) > 0:
            for device_id in device_ids:
                tmp = torch.device(f"cuda:{device_id}")
                print(f"{'Choosen device':<16}: {tmp}")
                print(f"{'Device name':<16}: {torch.cuda.get_device_name(tmp)}")
        else:
            print(f"{'Choosen device':<16}: {devices[0]}")
        print("------------------------------------------\n")

    return devices


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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        # NOTE: patience (int): How long to wait after last time validation loss improved.
        #                       Default: 7
        #       verbose (bool): If True, prints a message for each validation loss improvement.
        #                       Default: False
        #       delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
        #                       Default: 0

        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        path = os.path.join(self.save_path, "checkpoint.pt")
        torch.save(model.state_dict(), path)  # save best model state dict
        self.val_loss_min = val_loss


def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    dl_utils.plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else dl_utils.plt.gca()

    def has_one_axis(X):
        """如果X有一个轴，输出True"""
        return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


if __name__ == "__main__":
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f"{timer.stop():.5f} sec")

    timer.start()
    d = a + b
    print(f"{timer.stop():.5f} sec")
