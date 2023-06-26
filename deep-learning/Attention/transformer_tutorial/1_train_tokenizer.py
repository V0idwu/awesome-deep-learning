from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])

# 按不可见字符进行拆分（去除不可见字符）
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# UNK： 未登录词
# CLS：一般放在句首，用于分类任务
# SEP：分隔符，用于分隔长句子
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

tokenizer.train(["Attention/transformer_tutorial/data/三体.txt"], trainer=trainer)
tokenizer.save("Attention/transformer_tutorial/model_save/my_tokenizer.json")
