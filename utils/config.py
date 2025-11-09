import os
import torch
from tokenizers import Tokenizer


class Config:
    # ---- Paths ----
    # Assume the script is run from the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tokenizer.json")

    # ---- Tokenizer ----
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    # ---- Special token IDs (from tokenizer.json) ----
    pad_token_id = tokenizer.token_to_id("[PAD]")    # 2
    bos_token_id = tokenizer.token_to_id("[S]")      # 0
    eos_token_id = tokenizer.token_to_id("[/S]")     # 1
    joke_token_id = tokenizer.token_to_id("[JOKE]")  # 6

    # Safety checks (helpful during debugging)
    assert pad_token_id is not None,  "PAD token [PAD] not found in tokenizer"
    assert bos_token_id is not None,  "BOS token [S] not found in tokenizer"
    assert eos_token_id is not None,  "EOS token [/S] not found in tokenizer"
    assert joke_token_id is not None, "JOKE token [JOKE] not found in tokenizer"

    # ---- Model hyperparameters ----
    max_seq_len = 256   # Matches tokenizer truncation setting
    d_model = 256
    n_heads = 8
    d_ff = 1024
    n_layers = 4
    dropout = 0.1

    # ---- Training hyperparameters ----
    batch_size = 32
    lr = 3e-4
    weight_decay = 0.01
    num_epochs = 5

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"