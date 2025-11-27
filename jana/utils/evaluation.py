import re
from transformers import pipeline


def topic_inclusion(prompt, joke):
    topics = re.findall(r"about (.*)", prompt)
    if not topics:
        return 0
    topics = topics[0].split(" and ")
    included = [t for t in topics if t.lower() in joke.lower()]
    return len(included) / len(topics)

def check_humour(joke):
    pipe = pipeline("text-classification", 
                    model="mohameddhiab/humor-no-humor")
    return pipe(joke)[0]


TOPIC_KEYWORDS = {
    "elephants": {"elephant", "trunk", "tusk"}
}

def keyword_hit_rate(generated_text: str, topic: str) -> float:
    words = set(re.findall(r"\w+", generated_text.lower()))
    keys = TOPIC_KEYWORDS.get(topic, set())
    if not keys:
        return 0.0
    hits = sum(1 for k in keys if k in words)
    return hits / len(keys)

