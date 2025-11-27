import torch.nn as nn

class EncoderDecoderTransformer(nn.Module):

    def __init__(self, vocab_size=0, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        pass
    
    def forward(self):
        pass
    
    def next_token(self, context: list[str], topic) -> str:
        if not context:
            return "This"
        if context[-1] == "This":
            return "is"
        if context[-1] == "is":
            return "a"
        if context[-1] == "a":
            return "joke"
        if context[-1] == "joke":
            return "about"
        if context[-1] == "about":
            return "an"
        if context[-1] == "an":
            return topic
        return " "
    
    def generate(self, tokenizer, prompt, max_len=10) -> str:
        ctx, out = [], []
        for _ in range(max_len):
            token = self.next_token(ctx, topic=prompt)
            out.append(token)
            ctx.append(token)
        return " ".join(out)