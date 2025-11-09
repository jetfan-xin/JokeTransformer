# 🤖 Decoder-only Transformer

Build and train (small datasets completed, whole dataset TBD) a small decoder-only transformer on CPU hardware to balance efficiency, creativity, and interpretability.

---

## 🧩 Project Structure

```shell
.
├── README.md                       # Project description and documentation
├── DECODER_ONLY.md                 # 🆕 Add: Decoder-only model realization description
├── best_decoder_only.pt            # 🆕 Generated: Save checkpoint of the best trained model (TBD)
│
├── data/
│   ├── add_topics.py               # Add topic keywords using POS tagging (spaCy)
│   ├── clean.py                    # Clean and normalize raw joke datasets
│   ├── combine_datasets.py         # Merge multiple open-source joke datasets
│   ├── jokes_tiny.txt              # Small example dataset (for quick testing)
│   ├── processed/
│   │   ├── tokenizer.json          # Trained BPE tokenizer (52k vocab)
│   │   ├── final_combined_jokes.csv# Combined and cleaned full dataset
│   │   ├── train.csv               # Training split
│   │   ├── val.csv                 # Validation split
│   │   └── test.csv                # Test split
│   ├── split_dataset.py            # 🆕 Add: Split the full dataset (90:5:5)
│   ├── split_tiny_dataset.py       # 🆕 Add: Create a small debug set (80:10:10)
│   └── topic_keyword.json          # Extracted topic nouns from jokes
│
├── main.py                         # 🆕 Add: Interactive inference (generate jokes by topic)
├── train.py                        # 🆕 Add: Model training pipeline (for decoder-only transformer)
│
├── models/
│   └── decoder_only.py             # 🆕 Add: Implementation of the decoder-only transformer model
│
├── utils/
│   ├── config.py                   # 🆕 Add: Global configuration (paths, tokenizer, hyperparameters)
│   ├── dataset.py                  # 🆕 Add: Dataset and dataloader definitions
│   └── inference.py                # 🆕 Add: Joke generation and decoding functions
│
├── requirements.txt                # Dependencies
├── notebooks/                      # Jupyter notebooks (experiments & visualization)
├── configs/                        # Optional external config files
├── src/                            # (reserved for extensions or alternate architectures)
└── test/                           # Unit and integration tests
```

⸻

## 🧠 Model Architecture: Decoder-Only Transformer

The core model is implemented in `models/decoder_only.py` as the class:

```python
class DecoderOnlyTransformer(nn.Module):
    ...
```

### ⚙️ Key Idea

A decoder-only transformer predicts the next token based on all previous tokens — this is known as autoregressive next-token prediction.

- Input example:

```
[S] Tell me a joke about cats and rain [JOKE] Why did the cat bring an umbrella? ...
```

- During training:
    - The model sees the entire sequence (prompt + joke).
    - The loss is computed only on the joke portion (after [JOKE]).
    - This ensures it learns to generate funny continuations while respecting the topic.

---

### 🧱 Model Components

The implementation uses pure PyTorch — no prebuilt transformer modules.

| Component | File | Description |
| --- | --- | --- |
| Embedding layers | tok_emb, pos_emb | Convert token IDs and positions to dense vectors |
| MultiHeadSelfAttention | MultiHeadSelfAttention class | Each token attends to all previous ones (causal masking) |
| FeedForward | Two-layer MLP with GELU | Expands and compresses features per token |
| DecoderBlock | Combines attention + feedforward + layer normalization | |
| Final LayerNorm + Linear Head | lm_head | Maps hidden states to vocabulary logits for next-token prediction |

Causal masking ensures the model cannot look ahead, preserving autoregressive generation.

---

### 🧩 Training Objective

In train.py, the model learns by minimizing the cross-entropy loss between predicted and true next tokens.

```python
loss = CrossEntropyLoss(ignore_index=-100)
```

Only tokens after [JOKE] are included in the loss via a loss_mask.

This isolates the joke generation task from the prompt text.

---

### 🧠 Efficiency Focus
- Model size: ~5M parameters
- Optimized for CPU training
- Low sequence length (≤256 tokens)
- Gradient clipping (clip_grad_norm_) to stabilize small-batch training
- Weight tying between embeddings and output layer (lm_head.weight = tok_emb.weight)

---

## 🏋️ Training Pipeline

File: train.py
1. Loads processed data (train.csv, val.csv).
2. Creates DataLoader using JokeDataset from utils/dataset.py.
3. Initializes the DecoderOnlyTransformer.
4. Trains for multiple epochs with AdamW optimizer.
5. Evaluates on validation set and saves the best model as best_decoder_only.pt.

Run:

```shell
python train.py
```

---

## 💬 Interactive Inference

**File:** main.py

Once trained, you can interactively generate jokes:

```shell
python main.py
```

** Example session:**

```
Using device: cpu
Enter topics for your joke (or 'quit' to exit): cats and rain

Generated joke:
Why did the cat bring an umbrella? Because it didn’t want a wet paw!
```

Under the hood, it calls:

```python
from utils.inference import generate_joke
```

which:

- Encodes prompt → [S] prompt [JOKE]
- Iteratively predicts next tokens (sampling with temperature/top-k)
- Stops at [EOS] or when max_new_tokens reached

⸻

## 🧮 Dataset Workflow

| Step | Script | Description |
| --- | --- | --- |
| 1 | combine_datasets.py | Merge multiple open-source joke datasets |
| 2 | clean.py | Normalize, deduplicate, filter |
| 3 | add_topics.py | Extract topic keywords using POS tagging |
| 4 | split_dataset.py | Split dataset 90:5:5 for training |
| 5 | split_tiny_dataset.py | Create 100-sample mini dataset for debugging |

---

## 🧰 Configuration

All settings are centralized in utils/config.py:

```python
d_model = 256
n_heads = 8
d_ff = 1024
n_layers = 4
dropout = 0.1
batch_size = 32
lr = 3e-4
num_epochs = 5
max_seq_len = 256
```

---

## 📊 Outputs

After training:

```
best_decoder_only.pt  # saved best model
data/processed/train.csv, val.csv, test.csv  # processed datasets
```

---

## 🧠 Summary

| Aspect | Description |
| --- | --- |
| Goal | Generate topic-conditioned jokes efficiently |
| Architecture | Decoder-only Transformer (GPT-style) |
| Training Data | 700K+ cleaned short jokes |
| Evaluation | Humor quality, relevance, linguistic fluency |
| Efficiency | CPU-friendly, small parameter count, low sequence length |