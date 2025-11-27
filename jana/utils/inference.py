import torch
import re

def generate_joke(model, tokenizer, prompt):
    topics = re.findall(r"about (.*)", prompt)
    topic = topics[0].split(" ")[0]
    joke = model.generate(tokenizer, topic)
    return joke
