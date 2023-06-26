import numpy as np
import torch
import torch.nn as nn
from model.model_config import ModelConfig
from model.transformer import Transformer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextClsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = Tokenizer.from_file("Attention/transformer_tutorial/model_save/my_tokenizer.json")
        with open(
            "Attention/transformer_tutorial/data/toutiao-text-classfication-dataset/toutiao_cat_data.txt", "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()

        index = 0
        tagMap = {}
        data = []
        for line in lines:
            ts = line.strip().split("_!_")
            if len(ts) != 5:
                continue

            id = ts[1]
            label = ts[2]
            text = ts[3]
            if not id in tagMap:
                tagMap[id] = {"index": index, "label": label}
                index += 1
            data.append([text, tagMap[id]["index"]])

        self.reMap = {}
        for id in tagMap:
            self.reMap[tagMap[id]["index"]] = tagMap[id]["label"]

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        ids = [2] + self.tokenizer.encode(text[:101]).ids + [3]
        if len(ids) < 103:
            ids = ids + [0 for _ in range(103 - len(ids))]

        return np.array(ids, dtype=np.int64), label


class TextClsModel(nn.Module):
    def __init__(self, config, label_count):
        super(TextClsModel, self).__init__()
        self.transformer = Transformer(config)
        self.f1 = nn.Linear(config.hidden_dim, label_count).to(config.device)

    def forward(self, x):
        p, _ = self.transformer(x)
        return self.f1(p)


def train():
    data = TextClsDataset()
    loader = DataLoader(data, batch_size=20, shuffle=True)

    config = ModelConfig(device=torch.device("cuda:0"))
    config.load("Attention/transformer_tutorial/model_save/config.json")

    model = TextClsModel(config, len(data.reMap)).to(config.device)
    model.transformer.load_state_dict(
        torch.load("Attention/transformer_tutorial/model_save/transformer.pt", map_location=config.device)
    )

    loss_function = nn.CrossEntropyLoss()
    epoch = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    step = 0

    pbar = tqdm(loader)
    for e in range(epoch):
        for ids, label in pbar:
            ids = ids.to(config.device)
            label = label.to(config.device)
            p = model(ids)
            optimizer.zero_grad()
            loss = loss_function(p, label)
            loss.backward()
            optimizer.step()
            step += 1

            err = 0
            pLabel = torch.argmax(p, dim=1)
            for i in range(pLabel.size(0)):
                if int(pLabel[i]) != int(label[i]):
                    err += 1
            acc = 1
            if err != 0:
                acc = 1 - err / pLabel.size(0)
            desc = f"[{e+1}/{epoch}][{step}][{float(loss):.4f}][{acc:.4f}]"
            pbar.set_description(desc)


if __name__ == "__main__":
    train()
