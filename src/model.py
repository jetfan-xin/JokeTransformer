class ToyJokeModel:
    def __init__(self, topics_dict: dict[str, set[str]]):
        self.topics_dict = topics_dict

    def next_token(self, context: list[str]) -> str:
        if not context:
            return "Why"
        if context[-1] == "Why":
            return "did"
        if context[-1] == "did":
            return "the"
        if context[-1] == "the":
            return "elephant"
        return "laugh?"