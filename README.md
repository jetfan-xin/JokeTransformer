# uhh-ias-ml
Machine Learning Project (UHH - IAS)

#  Joke-LLM — Topic-Conditioned Joke Generator  
**Project Sketch**

A small LLM that generates **short jokes** about a given **topic**, trained from scratch using a **tiny decoder-only Transformer**.  
Uses the *Short Jokes* dataset (subset).  
Later, we’ll compare it with a simpler **RNN baseline** to understand trade-offs.

##  Project Pipeline
User topic → Preprocess data → Tokenize → Train encoder-only tranformer → Generate → Evaluate

### 1.Data
- **Dataset:** Short Jokes (on Hugging Face) — use 10 000 jokes.  
- **Cleaning:** lowercase, strip emojis, filter short/long lines, etc...  
- **Format:**  <TOPIC> {topic} <SEP> <BOS> {joke} <EOS>
- **Topics:** keyword dictionary (e.g., `"elephants" → {"elephant","trunk","tusk"}`).


### 2. Tokenization
- **Type:** Subword eg.(BPE)  
- **Vocab size:** 8 000  
- **Special tokens:** `<TOPIC> <SEP> <BOS> <EOS> <PAD>`


### 3. Model (Decoder-Only Transformer)


### 4.Training




### 5. Decoding (Generation)
Input prompt: <TOPIC> elephants <SEP> <BOS>
Strategies:
- **Greedy:** always pick most probable token (baseline)  
- **Top-p (0.9) + Temperature (0.8 – 1.0):** adds controlled randomness  
- **No-repeat-trigram:** avoids repetition



### 6.Evaluation Metrics
| Aspect | Metric | Type |
|---------|---------|------|
| **Accuracy** | Keyword hit rate / semantic similarity | automatic |
| **Creativity** | Distinct-n / novelty (embedding distance) | automatic |
| **Fluency** | perplexity | ? |
| **Humor** | Human rating | ? |




### 7.RNN Baseline
Later → train a tiny RNN (1 layer LSTM/GRU, 128 hidden) on the same data  
to compare loss, topicality & funny-score vs. Transformer.

