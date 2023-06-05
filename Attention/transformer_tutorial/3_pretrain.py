import random

import numpy as np
import torch
import torch.optim as optim
from model.model_config import ModelConfig
from model.pretrain_model import PretrainModel
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PretrainDataSet(Dataset):
    def __init__(self) -> None:
        super(PretrainDataSet, self).__init__()
        self.tokenizer = Tokenizer.from_file("Attention/transformer_tutorial/model_save/my_tokenizer.json")

        with open("Attention/transformer_tutorial/data/三体.txt", "r") as f:
            lines = f.readlines()

        ds = []
        for line in tqdm(lines):
            line = line.strip()
            if line == "":
                continue
            ts = line.split("。")
            if len(ts) < 2:
                continue
            for i in range(len(ts) - 1):
                ds.append([ts[i], ts[i + 1]])

        nds = []
        for i in range(len(ds)):
            id1 = random.randint(0, len(ds) - 1)
            id2 = random.randint(0, 1)
            nds.append([ds[i][0], ds[id1][id2]])

        self.data = []
        for d in ds:
            self.data.append(d + [1])
        for d in nds:
            self.data.append(d + [0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        ids1 = self.tokenizer.encode(d[0][:50]).ids
        ids2 = self.tokenizer.encode(d[1][:50]).ids
        ids = ([2] + ids1 + [3] + ids2 + [3])[:103]

        padding_size = 103 - len(ids)
        token_type_ids = [0 for _ in range(len(ids1) + 2)] + [1 for _ in range(len(ids2) + 1)]
        mask = np.random.random(len(ids))
        mask = (mask > 0.05).astype(np.int64)
        if padding_size > 0:
            mask = np.hstack((mask, np.ones(padding_size)))
            ids += [1 for _ in range(padding_size)]
            token_type_ids += [0 for _ in range(padding_size)]

        mask_ids = []
        for i in range(len(ids)):
            if mask[i] == 1:
                mask_ids.append(ids[i])
            else:
                mask_ids.append(4)

        return (
            np.array(ids, dtype=np.int64),
            np.array(token_type_ids, dtype=np.int64),
            np.array(mask_ids, dtype=np.int64),
            d[2],
        )


def pretrain():
    config = ModelConfig(device=torch.device("cuda"))
    config.load("Attention/transformer_tutorial/model_save/config.json")

    model = PretrainModel(config)

    data = PretrainDataSet()
    loader = DataLoader(data, batch_size=20, shuffle=True)
    epoch = 20
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    step = 0

    pbar = tqdm(loader)
    for e in range(epoch):
        for ids, token_type_ids, mask_ids, ns in pbar:
            ids = ids.to(config.device)
            token_type_ids = token_type_ids.to(config.device)
            mask_ids = mask_ids.to(config.device)
            ns = ns.to(config.device)

            optimizer.zero_grad()
            loss = model(mask_ids, token_type_ids, ids, ns)
            loss.backward()
            optimizer.step()
            step += 1
            desc = f"[{e+1}\{epoch}][{step}][{float(loss)}]"
            pbar.set_description(desc)

            if step % 100 == 0:
                wmp, cls = model(mask_ids, token_type_ids)
                pids = torch.argmax(wmp, dim=2)
                for i in range(pids.size(0)):
                    s = data.tokenizer.decode(pids.detach().cpu().numpy()[i])
                    ori_s = data.tokenizer.decode(ids.detach().cpu().numpy()[i])
                    print(f"\n\n==[{s}]==\n==[{ori_s}]==\n\n")
            if step % 5000 == 0:
                torch.save(model.state_dict(), f"Attention/transformer_tutorial/model_save/model_pretrain_{step}.pt")
    torch.save(model.state_dict(), f"Attention/transformer_tutorial/model_save/pretrain_done.pt")


if __name__ == "__main__":
    pretrain()
