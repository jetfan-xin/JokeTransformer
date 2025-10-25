import re

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


