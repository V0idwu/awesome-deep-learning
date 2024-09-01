import random
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

sys.path.insert(0, Path("").parent.parent.resolve().as_posix())
from utils import base_util


class LinearRegressionDataset(data.Dataset):
    """
    synthetic linear regression dataset
    """

    def __init__(self, w, b, num_examples):
        self.features, self.labels = self.synthetic_data(w, b, num_examples)

    def synthetic_data(self, w, b, num_examples):
        """y = Xw + b + noise"""
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

    def get_dataloader(self, batch_size):
        return data.DataLoader(data.TensorDataset(self.features, self.labels), batch_size, shuffle=True)

    def data_iter(self, batch_size):
        num_examples = len(self.features)
        indices = list(range(num_examples))
        # 这些样本是随机读取的，没有特定的顺序
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
            yield self.features[batch_indices], self.labels[batch_indices]

    def get_dataloader_scratch(self, batch_size):
        return self.data_iter(batch_size)

    def render(self):
        base_util.set_figsize()
        base_util.plt.scatter(self.features[:, 1].detach().numpy(), self.labels.detach().numpy(), 1)


class FashionMNISTDataset(data.Dataset):
    """
    FashionMNIST dataset
    """

    def __init__(self, root=Path("./data"), resize=None):
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        self.mnist_train, self.mnist_test = self.load_data_fashion_mnist(root, trans)

    def load_data_fashion_mnist(self, root, trans):
        """Downloads the Fashion-MNIST dataset and loads it into memory.
        Args:
            batch_size (int): The batch size for the data loader.
            trans (torchvision.transforms): The transformation to apply to the dataset.
        Returns:
            tuple: A tuple containing two data loaders - one for training data and one for testing data.
        """

        mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)

        return mnist_train, mnist_test

    def get_dataloader(self, batch_size):
        return (
            data.DataLoader(self.mnist_train, batch_size, shuffle=True, num_workers=base_util.get_dataloader_workers()),
            data.DataLoader(self.mnist_test, batch_size, shuffle=False, num_workers=base_util.get_dataloader_workers()),
        )

    def render(self, batch_size=18):
        X, y = next(iter(data.DataLoader(self.mnist_train, batch_size)))
        base_util.show_images(X.reshape(18, 64, 64), 2, 9, titles=self.get_fashion_mnist_labels(y))

    def get_fashion_mnist_labels(self, labels):
        """Returns the text labels of the Fashion-MNIST dataset.

        Args:
            labels (list): The numerical labels of the dataset.

        Returns:
            list: The corresponding text labels.

        Example:
            >>> labels = [0, 1, 2, 3, 4]
            >>> get_fashion_mnist_labels(labels)
            ['t-shirt', 'trouser', 'pullover', 'dress', 'coat']
        """

        text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        return [text_labels[int(i)] for i in labels]


if __name__ == "__main__":
    batch_size = 10
    num_examples = 1000
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    dataset = LinearRegressionDataset(true_w, true_b, num_examples)
    # dl_utils.set_figsize()
    # dl_utils.plt.scatter(dataset.features[:, 1].detach().numpy(), dataset.labels.detach().numpy(), 1)
    data_iter = dataset.get_dataloader_scratch(batch_size)
    print(next(iter(data_iter)))

    dataset = FashionMNISTDataset(resize=64)
    train_iter, test_iter = dataset.get_dataloader(batch_size=32)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
    # dataset.render()