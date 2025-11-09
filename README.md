
<p  align="center"><img  src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjYwdnJldTl6MzRocTV3b3d1bXlud2k3emM3OG9lZmtwZTI0amNkeiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wKSpqpmltIyPsIA1IM/giphy.gif"
alt="Alt Text" width="200"/></p>

  

# Efficient Joke Transformer

*A Master’s Project on Efficient Methods of Machine Learning — University of Hamburg*

  

---

  

## Abstract

Humor is a fundamental expression of intelligence — capturing cultural, linguistic, and contextual nuances.

In this project, we explore **efficient transformer architectures** capable of generating jokes given a user-provided topic or set of nouns (e.g., *“Tell me a joke about elephants and rain”*).

  

Unlike large pretrained LLMs, our model is **trained entirely from scratch** on a limited dataset and constrained hardware (CPU / laptop-based setup).

This emphasizes **architectural and algorithmic efficiency** — using smart model design, dataset preprocessing, and optimization techniques to achieve quality results under strict computational limits.

  

The project compares **two transformer variants**:

1.  **Decoder-only Transformer**

2.  **Encoder–Decoder Transformer**

  

Both are evaluated for humor quality, topic relevance, and linguistic correctness, while maintaining comparable model sizes to ensure fair comparison.

  

---

  

## Repository Structure

  

```

efficient-joke-transformer/

│

├── data/ # Data storage

│ ├── raw/ # Original downloaded dataset

│ ├── processed/ # Cleaned + tokenized datasets

│ └── dataset_loader.py # Dataset loading utilities

│

├── models/ # Transformer architectures

│ ├── decoder_only.py # GPT-style model

│ ├── encoder_decoder.py # Seq2Seq model

│ └── __init__.py

│

├── utils/ # Helper functions and pipelines

│ ├── preprocessing.py # Cleaning, tokenization, PoS tagging

│ ├── train_utils.py # Training loops, optimizer setup, checkpointing

│ ├── evaluation.py # Metrics: humor, topic inclusion, grammar

│ ├── inference.py # Inference (generate jokes from prompts)

│ └── config.py # Configurations (hyperparameters, paths)

│

├── notebooks/ #  Exploratory analysis / visualizations

│ ├── data_exploration.ipynb

│ └── model_playground.ipynb

│

├── tests/ #  Testing modules

│ ├── test_data_loading.py

│ ├── test_generation.py

│ ├── test_evaluation.py

│ └── __init__.py

│

├── main.py # Main entry point (train/infer/eval CLI)

├── requirements.txt

└── README.md

  

````

  
  

---

  

##  Project Overview



  | **Aspect**          | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal**            | Generate jokes about given topics/nouns efficiently using custom-built transformers.                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Focus**           | Efficiency in architecture and training (CPU-friendly, few parameters).                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Datasets**        | [Short Jokes Dataset](https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes) (~200K jokes).<br>[Amirkid/jokes](https://huggingface.co/datasets/Amirkid/jokes).<br>[200k Short Texts for Humor Detection](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection?resource=download).<br>[rJokesData train.tsv.gz](https://github.com/orionw/rJokesData/blob/master/data/train.tsv.gz).<br>[shuttie/dadjokes](https://huggingface.co/datasets/shuttie/dadjokes). |
| **Models**          | (1) Decoder-only Transformer, (2) Encoder–Decoder Transformer.                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Evaluation**      | Humor, syntax/semantics, topic inclusion, human judgment.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Frameworks**      | PyTorch, HuggingFace Tokenizers, spaCy (for PoS tagging).                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Hardware Target** | Laptop / CPU-only training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |


##  Dataset & Preprocessing

  

###  Source

- Dataset: Short Jokes Dataset (~200K entries)

- Dataset: Amirkid/jokes (Hugging Face)

- Dataset: 200k Short Texts for Humor Detection (Kaggle)

- Dataset: rJokesData (train.tsv.gz on GitHub)

- Dataset: shuttie/dadjokes (Hugging Face)

- TOTAL COMBINED DATASET: 726,787 rows

  

###  Cleaning

1. **Remove** incomplete or offensive jokes (regex + profanity filtering).

2. **Normalize** text (lowercase, remove redundant punctuation).

3. **Remove duplicates** and extremely short jokes (< 5 words).

  


###  POS Tagging & Topic Conditioning

To improve topic relevance and semantic clarity:

- Use `spaCy` POS tagger to extract **nouns** and **verbs**.  
- These serve as potential *conditioning keywords* (topics).  
- Example transformation:

| **Original Joke** | **Condition** | **Model Input** |
|--------------------|----------------|------------------|
| “Why did the chicken cross the road? To get to the other side!” | `chicken`, `road` | `<prompt> Tell me a joke about chicken and road <prompt> <joke> Why did the chicken cross the road? To get to the other side! <joke>` |


---

##  Model Architectures

  

### 1️ Decoder-Only Transformer (GPT-style)

This version treats the entire joke generation as a **language modeling** task.

  

**Input Format:**

  ```
  <prompt> Tell me a joke about elephants and rain <prompt> <joke> Why did the elephant bring an umbrella? Because he didn't want a wet trunk. <joke>
  ```


**Training Objective:**

Predict the next token given previous context.

  

**Advantages:**

- Simpler, smaller model (efficient for CPU training).

- Easier autoregressive generation.

  

**Challenges:**

- Prompt and joke share same space → may conflate prompt/joke boundaries.



### Encoder–Decoder Transformer (Seq2Seq)

The encoder reads the **prompt**, while the decoder generates the **joke**.

  

**Input / Output Example:**

- Encoder Input: `"Tell me a joke about elephants and rain"`

- Decoder Output: `"Why did the elephant bring an umbrella? Because he didn't want a wet trunk."`

  

**Advantages:**

- Clear separation between conditioning (prompt) and generation (joke).

- Can focus decoder on creative response.

  

**Challenges:**

- Slightly more parameters.

- More complex training loop.

  

---


### Fair Comparison

To ensure a **scientifically valid comparison**, both models will be:

- Matched by **parameter count** (≈ same embedding and hidden dimensions).  
- Trained on the **same dataset split** and **epochs**.  
- Evaluated on **identical metrics**.  

**Example:**

| **Attribute** | **Decoder-only** | **Encoder–Decoder** |
|----------------|------------------|---------------------|
| **Params** | ~5M | ~5M |
| **Training Time (CPU)** | Faster | Slightly slower |
| **Output Diversity** | High | Moderate |
| **Conditioning Strength** | Weaker | Stronger |


---


##  Training & Efficiency

  

- **Batch size:** small (e.g., 16)

- **Sequence length:** truncated to 64–128 tokens

- **Optimizer:** TBD

- **Early stopping** and **checkpointing** to save compute

---



##  **Evaluation**

A comprehensive evaluation protocol is used to assess both **model quality** and **efficiency**.  
We combine **quantitative metrics** (automatic, reproducible) and **qualitative evaluations** (human judgment) to ensure a fair comparison between the **Decoder-only** and **Encoder–Decoder** architectures.


### **Evaluation Overview**

| **Category** | **Metric / Method** | **Description** | **Type** | **Reference / Tool** |
|---------------|--------------------|-----------------|-----------|-----------------------|
| **Humor Quality** | **Human Evaluation (Blind A/B)** | Human judges rate two systems (decoder-only vs encoder–decoder) on 5-point scales: Funniness, Relevance, Coherence, Originality, and Safety. | Qualitative | Merrill et al. (2024), Li et al. (2016) |
|  | **Humor Classifier Score** | Probability of being humorous from pretrained humor detector. | Quantitative | [mohameddhiab/humor-no-humor](https://huggingface.co/mohameddhiab/humor-no-humor), Kaggle Humor Dataset |
| 🎭 **Creativity & Diversity** | **n-Novelty** | Fraction of generated n-grams unseen in the training set (1 − memorization rate). Measures originality. | Quantitative | Merrill et al. (2024) |
|  | **Distinct-1 / Distinct-2** | Ratio of unique unigrams/bigrams to all tokens — measures lexical diversity. Higher = more diverse. | Quantitative | Li et al. (2016) |
|  | **Self-BLEU-2 / 3** | Average BLEU similarity among generated jokes — lower = less mode collapse, more creative variety. | Quantitative | Zhu et al. (2018) |
|  **Linguistic Quality** | **Perplexity** | Measures fluency and syntactic quality of generated jokes using a reference LM. | Quantitative | GPT-2 small or KenLM |
|  | **Grammar / Syntax Checker** | Grammar correctness measured using LanguageTool or spaCy grammar checks. | Quantitative | LanguageTool |
|  **Topic Relevance** | **Keyword Inclusion** | Ensures that prompt nouns/topics appear in the generated joke text. | Quantitative | Keyword matching or semantic similarity |


###  **Humor Quality**

####  Human Evaluation (Blind A/B)
We conduct a **blind test** where human annotators compare outputs from:
- **Model A** → Decoder-only  
- **Model B** → Encoder–Decoder  

For 100 random prompts (e.g., “elephant”, “rain”, “school”), each rater scores both outputs on:

| **Criterion** | **Scale** | **Meaning** |
|----------------|------------|--------------|
| **Funniness** | 1–5 | 1 = not funny, 5 = hilarious |
| **Relevance** | 1–5 | How well the joke matches the given topics |
| **Coherence** | 1–5 | Logical sentence flow and readability |
| **Originality** | 1–5 | How novel the joke feels |
| **Safety** | OK / Not OK | Ensures no offensive or harmful content |

Each prompt gets **3 independent ratings**, and the **final score** is the averaged result per metric per model.

#### Humor Classifier
We automatically evaluate “funniness” using pretrained detectors:
- Primary: [**mohameddhiab/humor-no-humor**](https://huggingface.co/mohameddhiab/humor-no-humor)
- Backup: Kaggle Humor Classifier trained on 200k+ short jokes  

Output → *humor probability score (0–1)* for each generated joke.

---

### **Creativity & Diversity**

| **Metric** | **Purpose** | **Interpretation** |
|-------------|-------------|--------------------|
| **n-Novelty** | Checks originality of generated n-grams vs training set | High = creative |
| **Distinct-1 / 2** | Measures lexical variety | High = diverse vocabulary |
| **Self-BLEU-2 / 3** | Penalizes repetitive joke generation | Low = less mode collapse |

These metrics together evaluate whether the model merely memorizes the training jokes or truly generates new, varied, and interesting content.

---

### **Linguistic Quality**

We evaluate **fluency**, **grammar**, and **syntactic correctness**:
- **Perplexity** — Calculated using a small pretrained LM (e.g., GPT-2-small or KenLM).  
  → Lower perplexity = smoother and more natural sentences.
- **Grammar Checker** — Uses [**LanguageTool**](https://languagetool.org/) to detect grammatical/syntactic issues.  
  → Grammar accuracy = 1 − (#errors / #tokens).

---

### **Topic Relevance**

Ensures the generated jokes are relevant to the prompt:
- **Exact Keyword Match:** Verifies presence of all topic words.
- **Semantic Similarity:** Uses cosine similarity between topic embeddings (e.g., Sentence-BERT or spaCy) and the joke text to capture synonyms or related expressions (e.g., *rainy* for *rain*).

Final score = weighted combination of both.

---

### **System Efficiency**

Since the project emphasizes **efficient training and inference**, we report:

| **Metric** | **Description** |
|-------------|-----------------|
| **Training Time / Epoch** | Time required to complete one epoch on CPU |
| **Throughput (tokens/sec)** | Tokens generated per second |
| **Parameter Count** | Ensures model size parity between architectures |
| **Memory Usage** | Peak memory allocation measured via `torch.profiler` |

This allows a fair comparison between the **two architectures** under equal computational constraints.



##  Testing

  

To ensure reproducibility:

  

-  **Unit tests** for each module (e.g., tokenizer, model, dataloader).

-  **Integration tests** to verify full pipeline (prompt → generation → evaluation).

-  **Automatic CI** on GitHub Actions (optional).

Example:
```
def test_joke_generation():
	joke = model.generate("Tell me a joke about bananas")
	assert "banana" in joke.lower()
	assert len(joke.split()) > 3
```

##  Inference Example

```
from models.decoder_only import DecoderOnlyTransformer

model = DecoderOnlyTransformer()
prompt = "Tell me a joke about cats and programming" 
print(model.generate(prompt))
```
**Output:**  
> "Why did the cat sit on the keyboard? It wanted to keep an eye on the mouse!"



## Current Progress — Running Example

To test the current implementation, please use the following input format:  
**"Tell me a joke about \<noun>"**

Example command:
```bash
python main.py --mode infer --arch encdec --prompt "Tell me a joke about elephants" --eval
```
Expected output:
```
 Joke: This is a joke about an elephants      
 Topic Inclusion Score: 1.00
 Humour or No Humour  : NO_HUMOR
 NO_HUMOR Score         : 0.99
```
