import json
import random
from torch.utils.data import Dataset

class JokeDataset(Dataset):
    """
    Jokes dataset loader
    """
    def __init__(self, file_path, tokenizer, max_len=64):
        with open(file_path, "r") as f:
            self.jokes = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.jokes)