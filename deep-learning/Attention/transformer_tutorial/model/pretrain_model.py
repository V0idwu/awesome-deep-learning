import torch
import torch.nn as nn

from .pretrain_head import LMPredictHead, SCLsHead
from .transformer import Transformer


class PretrainModel(nn.Module):
    def __init__(self, config) -> None:
        super(PretrainModel, self).__init__()
        self.transformer = Transformer(config)

        self.wm = LMPredictHead(config)
        self.scls = SCLsHead(config)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)

        self.config = config

    def forward(
        self, input_ids, token_type_ids=None, mask_lm_labels=None, next_sentance_labels=None
    ):  # input_ids [batch_size, seq_len]
        p, e = self.transformer(input_ids, token_type_ids)

        wmp = self.wm(e)  # [batch_size, seq_len, vocab_size]
        sclsp = self.scls(p)  # [batch_size, 2]

        if mask_lm_labels is not None and next_sentance_labels is not None:
            loss1 = self.loss_function(wmp.view(-1, self.config.vocab_size), mask_lm_labels.reshape(-1))
            loss2 = self.loss_function(sclsp.view(-1, 2), next_sentance_labels.view(-1))
            return loss1 + loss2

        return wmp, sclsp
