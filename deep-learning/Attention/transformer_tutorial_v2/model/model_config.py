import json

import torch


class ModelConfig(object):
    def __init__(
        self,
        device=torch.device("cpu"),
        vocab_size=None,
        hidden_dim=None,
        max_len=None,
        n_head=None,
        drop_out=None,
        layer_count=None,
        encoder_layer_count=None,
        decoder_layer_count=None,
        padding_id=None,
    ) -> None:
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.n_head = n_head
        self.drop_out = drop_out
        # self.layer_count = layer_count
        self.encoder_layer_count = encoder_layer_count
        self.decoder_layer_count = decoder_layer_count
        self.padding_id = padding_id
        self.use_gpu = self.device == torch.device("cuda")

    def save(self, path):
        d = {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "max_len": self.max_len,
            "n_head": self.n_head,
            "drop_out": self.drop_out,
            "encoder_layer_count": self.encoder_layer_count,
            "decoder_layer_count": self.decoder_layer_count,
            "padding_id": self.padding_id,
        }
        d = json.dumps(d)
        with open(path, "w") as f:
            f.write(d)

    def load(self, path):
        with open(path, "r") as f:
            d = f.read()
        d = json.loads(d)
        self.vocab_size = d["vocab_size"]
        self.hidden_dim = d["hidden_dim"]
        self.max_len = d["max_len"]
        self.drop_out = d["drop_out"]
        self.n_head = d["n_head"]
        # self.layer_count = d["layer_count"]
        self.encoder_layer_count = d["encoder_layer_count"]
        self.decoder_layer_count = d["decoder_layer_count"]
        self.padding_id = d["padding_id"]
