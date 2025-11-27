#!/usr/bin/env python3
"""
Quick tester for eval_metrics.py

- Feeds a few fake jokes + topics to each metric
- Prints the results in a readable way

Run with:
    python test_eval_metrics.py
"""

from typing import List, Dict
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from eval.metrics import (
    topic_recall,
    topic_soft_recall,
    gpt2_perplexity,
    max_bleu_to_training,
    is_copied_from_training,
    encode_sentences,
    max_embedding_similarity_to_training,
    is_semantic_duplicate,
    diversity_metrics,
)


def build_fake_data():
    # these are training jokes
    training_jokes: List[str] = [
        "A man and a woman walk into a bar. The bartender says, 'Is this a joke?'",
        "Why did the chicken cross the road? To get to the other side.",
        "I told my dentist I broke my teeth on hard candy. He said, 'Stop eating the wrapper.'",
        "Cats are like music. It's foolish to explain their value to those who don't appreciate them.",
        "It was raining cats and dogs, so I stepped in a poodle.",
        "It was raining so hard that my cat finally agreed to take a bath by accident."
    ]

    # Some fake generations from the model with associated requested topics
    generated_examples: List[Dict] = [
        {
            "id": "ex1",
            "requested_topics": ["man", "woman", "bar"],
            "generated_joke": (
                "A guy and a lady walk into a pub."
                "The bartender laughs and says, 'You two again?'"
            ),
        },
        {
            "id": "ex2",
            "requested_topics": ["chicken", "road"],
            "generated_joke": (
                "The chicken didn't cross the street, "
                "it just ordered a taxi and complained about traffic."
            ),
        },
        {
            "id": "ex3",
            "requested_topics": ["dentist", "tooth"],
            "generated_joke": (
                "My dentist told me to floss more, "
                "so I started bragging about my achievements."
            ),
        },
        {
            "id": "ex4",
            "requested_topics": ["cat", "rain"],
            "generated_joke": (
                "It was raining so hard that my cat "
                "finally agreed to take a bath by accident."
            ),
        },
        {
            "id": "ex5",
            "requested_topics": ["elephant", "friend"],
            "generated_joke": (
                "My friend bought an elephant for his room. "
                "I said, 'Where are you going to keep it?' "
                "He said, 'It won't fit in the closet, obviously.'"
            ),
        },
    ]

    return training_jokes, generated_examples


def main():
    training_jokes, generated_examples = build_fake_data()

    print("=" * 80)
    print("Building training embeddings for embedding-based similarity...")
    print("=" * 80)
    train_embeddings = encode_sentences(training_jokes)
    print(f"Encoded {len(training_jokes)} training jokes.\n")

    # Evaluate each generated joke
    for ex in generated_examples:
        ex_id = ex["id"]
        topics = ex["requested_topics"]
        joke = ex["generated_joke"]

        print("=" * 80)
        print(f"Example: {ex_id}")
        print("-" * 80)
        print(f"Requested topics: {topics}")
        print(f"Generated joke:\n{joke}\n")

        # 1) Topic Recall (hard)
        hard_recall, hard_full_hit = topic_recall(joke, topics)

        # 2) Topic Soft Recall (soft similarity using spaCy vectors)
        soft_recall, soft_full_hit = topic_soft_recall(joke, topics)

        # 3) GPT-2 Perplexity (fluency)
        ppl = gpt2_perplexity(joke)

        # 4) Max BLEU vs training jokes (surface-level copying)
        max_bleu = max_bleu_to_training(joke, training_jokes, max_refs=5)
        copied_flag = is_copied_from_training(max_bleu, threshold=0.8)

        # 5) Max embedding similarity vs training jokes (semantic duplication)
        max_sim = max_embedding_similarity_to_training(joke, train_embeddings)
        sem_dup_flag = is_semantic_duplicate(max_sim, threshold=0.9)

        # Print metrics for this example
        print("Topic metrics:")
        print(f"  Hard topic recall     : {hard_recall:.3f}")
        print(f"  Hard full hit (0/1)   : {hard_full_hit}")
        print(f"  Soft topic recall     : {soft_recall:.3f}")
        print(f"  Soft full hit (0/1)   : {soft_full_hit}")
        print()
        print("Fluency / language metric:")
        print(f"  GPT-2 perplexity      : {ppl:.3f}")
        print()
        print("Copying / memorization metrics:")
        print(f"  Max BLEU to training  : {max_bleu:.3f}")
        print(f"  Copied flag (BLEU>=0.8): {copied_flag}")
        print(f"  Max emb sim to train  : {max_sim:.3f}")
        print(f"  Semantic dup flag (sim>=0.9): {sem_dup_flag}")
        print()

    # Diversity metrics over all generated jokes
    print("=" * 80)
    print("Diversity over all generated jokes")
    print("=" * 80)
    all_generated_texts = [ex["generated_joke"] for ex in generated_examples]
    div = diversity_metrics(all_generated_texts)
    print(f"Distinct-1: {div['distinct_1']:.3f}")
    print(f"Distinct-2: {div['distinct_2']:.3f}")


if __name__ == "__main__":
    main()
