from tokenizers import Tokenizer
tok = Tokenizer.from_file("data/processed/tokenizer.json")

print(tok.encode("tell me a joke about cats").tokens)
print(tok.encode("tell me a joke about cats", "a cat walks into a bar").tokens)