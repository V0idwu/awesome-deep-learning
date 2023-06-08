from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("Attention/transformer_tutorial_v2/model_save/my_tokenizer.json")
s = "招财进宝财源广进，一本万利万事兴隆"
s = " ".join([c for c in s])
tokens = tokenizer.encode(s)

print(tokens.ids, tokens.tokens)
