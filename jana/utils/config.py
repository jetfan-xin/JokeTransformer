CONFIG = {
    #One place to tweak behavior without editing code; helps reproducibility
    "seed": 42,
    "batch_size": 4, #How many sequences we process in parallel
    "lr": 1e-4, #Learning rate how big of a step we take in the direction of the gradient
    "num_epochs": 10, #How many times we go through the entire dataset
    "seq_len": 256, #How many tokens we process at once
    # Jokes are short (1–2 sentences = ~30 tokens)
    "vocab_size": 30000,
    "d_model": 64, #Size of each token’s embedding vector (how many numbers represent a word)
    "n_heads": 4, #How many attention heads per block
    "n_layers": 4,#How many Transformer blocks we stack
    "dropout": 0.1, #How much dropout to apply
    "n_embd": 256, #Embedding dimension
}