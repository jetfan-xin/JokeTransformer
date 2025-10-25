## Architecture -- Transformers

#### (1) The Original Transformer

<img src="./images/encoderdecoder.png" alt="Original Transformer architecture" width="600"/>

Only used for **controlled joke generation** **with keyword-tags**.

1. Build a paired dataset:
   - **Input:** Prompt with extracted tags (elephant, trunk)
   - **Output:** original joke.

2. Train a small seq2seq model.

##### Advantage

More controllable, especially when compared with a small language model for text completion trained on small datasets without limitation, which could be too open to be precise.

##### Risk:

**Not sure whether the tag controll is too strong**: 

- Tag-conditioned generation risks losing creativity: Insert desired tags mechanically.

- It may do not really understand humor structure. E.g., setup+punchline+incongruity. But focus on the matching tags with content.

##### Solution:

**Add creative noise**: During training, occasionally shuffle or drop one tag, or add a random tag (5–10% of data), or replace a tag by its hypernymy (*Is-a* relationship). This forces the model to generalize instead of memorizing strict patterns.



#### (2) The Decoder-Only Transformer

https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse

<img src="./images/decoderonly" alt="Decoder-Only Transformer architecture" width="600"/>

**Modern generative LLMs** (GPT, Llama, etc.) are decoder-only.

It works for **autoregressive next-token prediction**. 

It drops the entire encoder (in other word, drop cross-attention layers). **It fits our scenario**:

- Original transformer is techinically used for sequence to sequence mapping task like translation. The only data source we have is a list of jokes, no sequence to sequence mapping included.

- Decoder-only transformer fits tasks that only need to do text completion, where the model generates text based on previous content.

- **Joke generation could also be a text completion task:

  With tag hints: generate tags related jokes: 

  > "Joke about elephants and trunks:" → model completion

  no encoder needed, just prepend the topic tokens (can also add creative noise).





# Topic Idea

#### Short Joke Generation via Encoder–Decoder Model v.s. Decoder-Only Model

##### A. Encoder–Decoder (seq2seq): Tags → Joke

- Input: “Joke about elephants and trunks” (1–4 tags)
- Output: joke line
- Training data: automatically built prompts from Short Jokes by keyword extraction (nouns/entities).
- Noise for generalization: drop a tag; add a plausible distractor tag; replace a tag by its hypernymy.



##### B. Decoder-Only (LM): Prompted Joke Completion

- Input: `Instruction: Tell a short funny joke about xx, xxx,xxx:\n`
- Output: generated joke continuation
- Training data: Mixed with synthetic “Instruction+Joke”.
- Noise for generalization: drop a tag; add a plausible distractor tag; replace a tag by its hypernymy.



Both models thus receive semantically equivalent guidance (topics) but through different architectures.

#### Is this comparison meaningful?

| **Dimension**       | **Encoder–Decoder (seq2seq)**           | **Decoder-Only (autoregressive)**          |
| ------------------- | --------------------------------------- | ------------------------------------------ |
| **Control**         | Strong (condition on tags)              | Medium (steer via prompt wording/keywords) |
| **Creativity**      | Risk of rigidity (copy tags, formulaic) | Higher spontaneity, better “voice”         |
| **Complexity/Cost** | Heavier (encoder + cross-attn)          | Lighter; faster to train                   |
| **Failure modes**   | Overfitting to tags, bland punchlines   | Off-topic or meandering jokes              |
| **Best use**        | Topic-targeted, keyword constraints     | Open-ended or softly conditioned humor     |





## Pipline

1. #### Data preprocessing

   - Dataset: 

     - shortjokes: https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes or 
     - ColBERT (humor marked with yes/no): https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection 

   - Split: train/val/test = 80/10/10

   - Tag extraction: noun extraction → keep 1–4 tags; lowercase; remove stopwords.

     > Tutorial(nltk): https://www.geeksforgeeks.org/nlp/unsupervised-noun-extraction-in-nlp/

   - Prompt templates for decoder only model: `"Jokes about [tag 1], [tag 2], ..., [tag n]: "`

2. #### Architecture hyperparameters

   - Match total parameter counts.

   - Tokenizer: byte pair encoding, which is used in modern LLMs like  GPT-2 to GPT-4, Llama 3, etc.  (E.g., BPE 8k. Tutorial: BPE from scratch: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)

3. #### Training

   Applicable training settings and tricks can be discussed later.

4. #### Evaluation

   ##### (1) Humor

   - Human judges: 

     - Blind A/B (A=seq2seq、B=decoder-only) on 100 prompts; 3 raters each.
     - Criteria: E.g., scores (1–5): Funniness, Relevance, Coherence, Originality; Safety (OK/Not OK). (refer to existed research later) 

     ```shell
     #Prototype#
     opic (keywords): elephant, trunk
     
     Option 1:
     "<joke text 1>"
     
     Option 2:
     "<joke text 2>"
     
     For each option, rate:
     - Funniness: 1 (not funny) … 5 (very funny)
     - Relevance: 1 (off-topic) … 5 (perfectly on-topic)
     - Coherence: 1 (broken/awkward) … 5 (clear & natural)
     - Originality: 1 (cliché/seen often) … 5 (fresh twist)
     - Safety: OK / Not OK
     
     Optional comment box (why you chose the scores)
     ```

   - Humor detector:

     - Huggingface: humor-no-humor

     - Alternative：Open-source humor detectors with impressive accuracy: https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection/code. 
       I'm afraid it is only able to evaluate jokes that generated based on ColBERT datasets, because both our joke generators and their detectors are small, Accordance of data source helps to enhance credibility.

   - Heuristic metrics about creativity / originality

     - n-Novelty: 1 − memorization rate by 4-gram Jaccard ≥ 0.6 against *training* jokes.

       > Merrill et al., 2024, *Evaluating n-Gram Novelty of Language Models Using RUSTY-DAWG*
       >
       > It measures the fraction of generated n-grams unseen in the training set.

     - Distinct-1/2: diversity on a 1k-sample set. Compared between training and generated joke sets.

       > Following Li et al. (2016), Distinct-n has become a standard metric for assessing lexical diversity and avoiding repetitive language in neural text generation.
       >
       > Li et al., 2016, A Diversity-Promoting Objective Function for Neural Conversation Models (NAACL)
       >
       > It computes the ratio of unique unigrams and bigrams to all tokens in 1 k generated samples (and optionally compared to the training set) — higher values indicate more varied word use.

     - Self-BLEU-2/3: lower is better (less mode collapse).

       > Proposed by Zhu et al. (2018) as a measure of mode collapse, Self-BLEU quantifies how similar generated samples are to each other, complementing Distinct-n.
       >
       > Zhu et al., 2018, Texygen: A Benchmarking Platform for Text Generation Models (SIGIR)
       >
       > It calculates BLEU-n scores between each generated joke and the rest of the generated set — lower scores mean higher diversity and less pattern repetition.

   ##### (2) Text quality

   - Relevance / Control: Percent of maintained tags
   - Fluency / Grammar: Perplexity

   ##### (3) System Efficiency

   - Throughput (token/sec) ....
