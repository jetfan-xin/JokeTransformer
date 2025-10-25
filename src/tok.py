# Simple

def tokenize(text: str):
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]

def detokenize(tokens):
    return " ".join(tokens)
