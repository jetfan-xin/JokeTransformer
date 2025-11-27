import torch

def generate_joke(model, tokenizer, prompt):
    joke = model.generate(tokenizer, prompt)
    return joke

