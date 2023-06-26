import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = torch.arange(0, 20)

    def __getitem__(self, index: int):
        x = self.data[index]
        y = x * 2
        return y

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
print(len(dataset))  # 20
print(dataset[3])  # tensor(6)


dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(len(dataloader))  # 20 / 4 = 5

for batch in dataloader:
    print(batch)
