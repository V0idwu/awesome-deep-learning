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
import platform
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from IPython import display
from matplotlib_inline import backend_inline
from torch.utils.tensorboard import SummaryWriter

base_util = sys.modules[__name__]


# NOTE: 为了保证实验的可重复性，需要设置随机数种子。
def setup_config(seed: int = 227):
    """set random seed for pytorch, numpy and random library."""
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


def try_cuda_gpu(i: int = 0) -> torch.device:
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_mps_gpu() -> torch.device:
    """Return mps device if exists, otherwise return cpu()."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# NOTE: CPU和GPU的设备设置
def try_gpu(gpu: bool = True, device_ids: list = ["0"], verbose: bool = False) -> list[torch.device]:
    if platform.system() == "Darwin":  # MacOS
        if torch.backends.mps.is_available():
            devices = [torch.device("mps")]
        else:
            devices = [torch.device("cpu")]

    elif platform.system() == "Linux" or platform.system() == "Windows":  # Linux or Windows

        if torch.cuda.is_available() and gpu:
            assert torch.cuda.device_count() > 0, "No GPU found, please run without --cuda"
            assert torch.cuda.device_count() >= len(
                device_ids
            ), f"Choosen GPU number {len(device_ids)} is larger than available GPU number {torch.cuda.device_count()}"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
            devices = [torch.device(f"cuda:{i}") for i in device_ids]
        else:
            devices = [torch.device("cpu")]

    if verbose:
        print("------------- Torch Lib Info -------------")
        print(f"{'Python version':<16}: {platform.python_version()}")
        print(f"{'PyTorch version':<16}: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"{'CUDA version':<16}: {torch.version.cuda}")
            print(f"{'cuDNN version':<16}: {torch.backends.cudnn.version()}")

        print("--------------- Device Info --------------")
        for device in devices:
            print(f"{'Choosen device':<16}: {device}")
            if torch.cuda.is_available() and gpu:
                print(f"{'Device name':<16}: {torch.cuda.get_device_name(device)}")
            else:
                # 输出CPU的信息
                print(f'{"Device name":<16}: {platform.processor()}')
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
    base_util.plt.rcParams["figure.figsize"] = figsize


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
    axes = axes if axes else base_util.plt.gca()

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


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = base_util.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        base_util.use_svg_display()
        self.fig, self.axes = base_util.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: base_util.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device=None):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if device is None:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_loss(net, data_iter, loss, device=None):  # @save
    """评估给定数据集上模型的损失"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if device is None:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        metric = Accumulator(2)  # 损失的总和,样本数量
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            out = net(X)
            y = y.reshape(out.shape[0])
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss_func, updater, timer, device):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        timer.start()
        # 计算梯度并更新参数
        y_hat = net(X)
        loss = loss_func(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            loss.backward()
            updater.step()
        else:
            # 自定义的权重优化器
            loss.sum().backward()
            updater(X.shape[0])
        with torch.no_grad():
            metric.add(float(loss.sum()), accuracy(y_hat, y), y.numel())
        timer.stop()
    return metric


def train(net, train_dataloader, test_dataloader, updater, loss_fn, num_epochs, device=None):
    """训练模型"""
    animator = Animator(xlabel="epoch", xlim=[1, num_epochs], legend=["train loss", "train acc", "test acc"])
    timer = Timer()
    if isinstance(net, torch.nn.Module):
        device = try_cuda_gpu()
        net.to(device)
    else:
        device = torch.device("cpu")

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_dataloader, loss_fn, updater, timer, device)
        train_loss, train_acc = train_metrics[0] / train_metrics[2], train_metrics[1] / train_metrics[2]
        # 返回训练损失和训练精度
        if test_dataloader is not None:
            test_acc = evaluate_accuracy(net, test_dataloader)
            # test_loss = evaluate_loss_gpu(net, test_dataloader, loss_func)
            animator.add(epoch + 1, (train_loss, train_acc, test_acc))
            print(f"train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")
        else:
            animator.add(epoch + 1, (train_loss, train_acc))
            print(f"train loss {train_loss:.3f}, train acc {train_acc:.3f}")
        print(f"{train_metrics[2] * num_epochs / timer.sum():.1f} examples/sec " f"on {str(device)}")


def copy_model_params(src_model, tar_model):
    """将模型tar_net的参数复制到模型src_net"""
    params1 = {param_name: param.data for param_name, param in src_model.named_parameters()}
    params2 = {param_name: param.data for param_name, param in tar_model.named_parameters()}
    params1.update(params2)
    src_model.load_state_dict(params1)


def get_k_fold_data(k, i, X, y):
    """NOTE: 折交叉验证在返回第i个切片作为验证数据，其余部分作为训练数据。"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(net, k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, verbose=False):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = net.copy()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        # if i == 0:
        #     dl_utils.plot(
        #         list(range(1, num_epochs + 1)),
        #         [train_ls, valid_ls],
        #         xlabel="epoch",
        #         ylabel="rmse",
        #         xlim=[1, num_epochs],
        #         legend=["train", "valid"],
        #         yscale="log",
        #     )
        if verbose:
            print(f"折{i + 1}，训练log rmse{float(train_ls[-1]):f}, " f"验证log rmse{float(valid_ls[-1]):f}")
    return train_l_sum / k, valid_l_sum / k


def init_weights(m):
    # NOTE: 初始化模型参数, net.apply(init_weights)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)
    elif type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.constant_(m.weight, val=1.0)
        torch.nn.init.constant_(m.bias, val=0.0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)


class TensorBoardLogger:
    def __init__(self, log_dir: str = "runs", comment: str = "") -> None:
        self.__log_dir = log_dir
        self.__comment = comment
        self.__writer = SummaryWriter(log_dir=self.__log_dir, comment=self.__comment)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.__writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int) -> None:
        self.__writer.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step)

    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None:
        self.__writer.add_histogram(tag=tag, values=values, global_step=global_step)

    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        self.__writer.add_image(tag=tag, img_tensor=img_tensor, global_step=global_step)

    def add_images(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        self.__writer.add_images(tag=tag, img_tensor=img_tensor, global_step=global_step)

    def add_figure(self, tag: str, figure: plt.Figure, global_step: int) -> None:
        self.__writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    def add_video(self, tag: str, vid_tensor: torch.Tensor, global_step: int) -> None:
        self.__writer.add_video(tag=tag, vid_tensor=vid_tensor, global_step=global_step)

    def add_audio(self, tag: str, snd_tensor: torch.Tensor, global_step: int, sample_rate: int = 44100) -> None:
        self.__writer.add_audio(tag=tag, snd_tensor=snd_tensor, global_step=global_step, sample_rate=sample_rate)

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        self.__writer.add_text(tag=tag, text_string=text_string, global_step=global_step)


# Define Alias

nn_Module = torch.nn.Module
ones_like = torch.ones_like
ones = torch.ones
zeros_like = torch.zeros_like
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
randn = torch.randn
matmul = torch.matmul
int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
sigmoid = torch.sigmoid
batch_matmul = torch.bmm
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)
