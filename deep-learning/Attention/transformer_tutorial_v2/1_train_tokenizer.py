import os

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

data_list = []
for root, paths, names in os.walk("Attention/transformer_tutorial_v2/data/couplet-clean-dataset/couplets"):
    for name in names:
        data_list.append(os.path.join(root, name))

tokenizer.train(data_list, trainer=trainer)
tokenizer.save("Attention/transformer_tutorial_v2/model_save/my_tokenizer.json")
