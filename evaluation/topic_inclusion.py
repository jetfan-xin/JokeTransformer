import os

def topic_exact_match(joke, topics):
    """Evaluate if the generated joke includes the exact input topics."""
    if not joke:
        print("No joke generated to evaluate.")
        return
        
    flag = 0
    for t in  topics.split(", "):
        if t in joke:
            print(f"Joke includes topic: {t}")
            flag = 1
    if flag == 0:
        print(f"Joke doesn't include any input topic\n")
    return


def topic_semiantic_similarity(joke, topics):
    """Evaluate if the generated joke is semantically similar to the input topics."""
    # Placeholder for semantic similarity evaluation logic
    return