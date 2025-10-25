from model import ToyJokeModel

TOPIC_TAGS = {"elephants": {"elephant", "trunk", "tusk"}}

def generate(topic: str, max_tokens: int = 12) -> str:
    model = ToyJokeModel(TOPIC_TAGS)
    ctx, out = [], []
    for _ in range(max_tokens):
        token = model.next_token(ctx)
        out.append(token)
        ctx.append(token)
    return " ".join(out)