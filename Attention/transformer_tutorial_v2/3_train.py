import os
import random

import numpy as np
import torch
import torch.optim as optim
from model.model_config import ModelConfig
from model.transformer import Transformer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CoupleDataset(Dataset):
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file("Attention/transformer_tutorial_v2/model_save/my_tokenizer.json")
        with open(os.path.join(path, "in.txt"), "r", encoding="utf-8") as f:
            in_data = f.readlines()
        with open(os.path.join(path, "out.txt"), "r", encoding="utf-8") as f:
            out_data = f.readlines()

        data = []
        for i in tqdm(range(len(in_data))):
            s, t = in_data[i][:30], out_data[i][:30]
            if len(s) == 0 or len(t) == 0:
                continue

            sids = self.tokenizer.encode(s).ids
            tids = self.tokenizer.encode(t).ids

            # A B C D
            # E F G H
            size = len(tids)
            for j in range(size):
                data.append([sids, [3] + tids[: size - j]])

            # [[A B C D], [3 E]]
            # [[A B C D], [3 E F]]
            # [[A B C D], [3 E F G]]
            # [[A B C D], [3 E F G H]]

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s, t = self.data[index]
        if len(s) < 30:
            s = s + [1 for _ in range(30 - len(s))]
        if len(t) < 31:
            t = t + [1 for _ in range(31 - len(t))]

        return np.array(s, dtype=np.int64), np.array(t, dtype=np.int64)


def train():
    config = ModelConfig(device=torch.device("cuda"))
    config.load("Attention/transformer_tutorial_v2/model_save/config.json")

    model = Transformer(config)

    data = CoupleDataset("Attention/transformer_tutorial_v2/data/couplet-clean-dataset/couplets/train")
    loader = DataLoader(data, batch_size=20, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_id)
    epoch = 20
    step = 0

    for e in range(epoch):
        pbar = tqdm(loader)
        for s, t in pbar:
            s = s.to(config.device)
            t = t.to(config.device)
            predict = model(s, t[:, :-1])
            predict_y = predict.contiguous().view(-1, predict.shape[-1])
            y = t[:, 1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = criterion(predict_y, y)
            loss.backward()
            optimizer.step()

            step += 1
            desc = f"[{e+1}/{epoch}][{step}][{float(loss):.4f}]"
            pbar.set_description(desc)

            if step % 100 == 0:
                pids = torch.argmax(predict, dim=-1)
                y = t[:, 1:]
                for i in range(predict.size(0)):
                    ss = data.tokenizer.decode(s.detach().cpu().numpy()[i])
                    ps = data.tokenizer.decode(pids.detach().cpu().numpy()[i])
                    oris = data.tokenizer.decode(y.detach().cpu().numpy()[i])
                    print(f"原始: {ss}\n预测: {ps}\n目标: {oris}\n")


if __name__ == "__main__":
    train()
