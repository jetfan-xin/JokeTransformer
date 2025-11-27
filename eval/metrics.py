
from typing import List, Tuple, Dict, Iterable
import math
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import spacy
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import spacy
nlp = spacy.load("en_core_web_md")  

# print("Tttttttttttttttesssssssssssttttttttttttttttt")
# doc = nlp("man woman bar cat dog")
# for token in doc:
#     print(token.text, np.linalg.norm(token.vector))
# Basic text preprocessing

def extract_content_nouns(text: str) -> List[str]:
    """
    Extract lemmatized nouns/proper nouns from text, lowercased,
    filtered for stopwords and very short tokens.

    This is useful for topic-based metrics.
    """
    doc = nlp(text)
    nouns = []
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and len(t.lemma_) >= 3:
            nouns.append(t.lemma_.lower().strip())
    return nouns

def topic_recall(
    generated_joke: str,
    requested_topics: List[str]
) -> Tuple[float, int]:
    """
    Compute topic recall and a full-hit flag.

    - requested_topics: list like ["man", "woman", "friend"]
    - generated_joke: the model output string

    Returns:
        recall: float in [0, 1]
        full_hit: 1 if all requested topics appear, else 0
    """
    # Normalize topics
    requested = [t.lower().strip() for t in requested_topics if t.strip()]
    if not requested:
        return 0.0, 0

    joke_nouns = extract_content_nouns(generated_joke)
    joke_set = set(joke_nouns)

    covered = 0
    for topic in requested:
        if topic in joke_set:
            covered += 1

    recall = covered / len(requested)
    full_hit = int(covered == len(requested))
    return recall, full_hit

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return float(np.dot(u, v) / denom)



#run: python -m spacy download en_core_web_md

def topic_soft_recall(
    generated_joke: str,
    requested_topics: List[str],
    sim_threshold: float = 0.4
) -> Tuple[float, int]:
    """
    Topic recall with soft matching using spaCy vectors.

    For each requested topic, we look for a noun in the joke that has
    cosine similarity >= sim_threshold with that topic's vector.

    Returns:
        soft_recall: fraction of topics 'covered' under similarity
        soft_full_hit: 1 if all topics covered, else 0
    """
    requested = [t.lower().strip() for t in requested_topics if t.strip()]
    if not requested:
        return 0.0, 0

    doc_joke = nlp(generated_joke)
    joke_nouns = [t for t in doc_joke if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]

    if not joke_nouns:
        return 0.0, 0

    # Precompute noun vectors and norms
    noun_vectors = np.array([t.vector for t in joke_nouns])
    noun_norms = np.linalg.norm(noun_vectors, axis=1)
    # Mask out any zero vectors
    valid_mask = noun_norms > 1e-8

    covered = 0
    for topic in requested:
        topic_doc = nlp(topic)
        if not topic_doc:
            continue
        topic_vec = topic_doc[0].vector
        topic_norm = np.linalg.norm(topic_vec)

        best_sim = 0.0
        if valid_mask.any() and topic_norm > 1e-8:
            # compute cosine similarity only on valid noun vectors
            valid_noun_vecs = noun_vectors[valid_mask]
            valid_norms = noun_norms[valid_mask]

            sims = (valid_noun_vecs @ topic_vec) / (valid_norms * topic_norm)
            best_sim = float(sims.max())

        if best_sim >= sim_threshold:
            covered += 1

    soft_recall = covered / len(requested)
    soft_full_hit = int(covered == len(requested))
    return soft_recall, soft_full_hit



# Load once
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()
gpt2_model.to("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def gpt2_perplexity(text: str) -> float:
    """
    Compute GPT-2 perplexity for a given text.

    Returns:
        perplexity: float, lower is better.
    """
    if not text.strip():
        return float("inf")

    enc = gpt2_tokenizer(
        text,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(gpt2_model.device)
    attention_mask = enc["attention_mask"].to(gpt2_model.device)

    outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # loss is mean over tokens (cross-entropy)
    loss = outputs.loss
    ppl = math.exp(loss.item())
    return ppl



_bleu_smoothing = SmoothingFunction().method1

def max_bleu_to_training(
    generated_joke: str,
    training_jokes: List[str],
    max_refs: int = None
) -> float:
    """
    Compute maximum BLEU score between generated_joke and a list of training_jokes.

    Args:
        generated_joke: candidate string
        training_jokes: list of reference joke strings
        max_refs: optional cap on number of refs to compare against
                  (e.g. for speed: randomly sample N references)

    Returns:
        max_bleu: float in [0, 1]
    """
    cand_tokens = generated_joke.split()
    if not cand_tokens or not training_jokes:
        return 0.0

    refs = training_jokes
    if max_refs is not None and len(training_jokes) > max_refs:

        import random
        refs = random.sample(training_jokes, max_refs)

    max_bleu = 0.0
    for ref in refs:
        ref_tokens = ref.split()
        bleu = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=_bleu_smoothing,
            weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
        )
        if bleu > max_bleu:
            max_bleu = bleu

    return max_bleu


def is_copied_from_training(max_bleu: float, threshold: float = 0.8) -> int:
    return int(max_bleu >= threshold)




# Load once
sent_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sent_model = SentenceTransformer(sent_model_name)

def encode_sentences(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of texts into sentence embeddings (numpy array).
    """
    embeddings = sent_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine similarity becomes dot product
    )
    return embeddings



def max_embedding_similarity_to_training(
    generated_joke: str,
    train_embeddings: np.ndarray
) -> float:
    """
    Compute maximum cosine similarity between generated joke embedding
    and precomputed training joke embeddings.

    Args:
        generated_joke: candidate joke
        train_embeddings: np.ndarray of shape (num_train, dim), L2-normalized.

    Returns:
        max_sim: float in [-1, 1] but usually [0, 1]
    """
    if train_embeddings.size == 0:
        return 0.0

    cand_emb = encode_sentences([generated_joke])[0]  # normalized if we used normalize_embeddings=True

    # cosine similarity reduces to dot product because of normalization
    sims = train_embeddings @ cand_emb
    max_sim = float(sims.max())
    return max_sim


def is_semantic_duplicate(max_sim: float, threshold: float = 0.9) -> int:
    return int(max_sim >= threshold)


from collections import Counter

def distinct_n(
    texts: Iterable[str],
    n: int
) -> float:
    """
    Compute distinct-n for a collection of texts.

    distinct_n = (# unique n-grams) / (# total n-grams)

    Args:
        texts: iterable of strings (generated jokes)
        n: n-gram size (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        distinct_n score in [0, 1]
    """
    ngram_counter = Counter()
    total_ngrams = 0

    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_counter[ngram] += 1
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    num_unique = len(ngram_counter)
    return num_unique / total_ngrams

def diversity_metrics(texts: Iterable[str]) -> Dict[str, float]:
    """
    Convenience wrapper: compute distinct-1 and distinct-2.
    """
    texts = list(texts)
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
    }
