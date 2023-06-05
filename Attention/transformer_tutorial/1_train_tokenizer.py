from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

tokenizer.train(["Attention/transformer_tutorial/data/三体.txt"], trainer=trainer)
tokenizer.save("Attention/transformer_tutorial/model_save/my_tokenizer.json")
