import re, json, spacy

nlp = spacy.load("en_core_web_sm")


def clean_joke(text):
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)
    return text.strip().lower()


def extract_topics(joke):
    doc = nlp(joke)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns[:2]  # limit to 2 nouns


def prepare_dataset(raw_path, processed_path):
    pass