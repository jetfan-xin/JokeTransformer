from utils.inference import generate_joke
from utils.tokenizer import Tokenizer
from models.decoder_only import DecoderOnlyTransformer
from models.encoder_decoder import EncoderDecoderTransformer
import torch

def test_decoder_generate_returns_text():
    tok = Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer().to(device)
    out = generate_joke(model, tok, 'Tell me a joke about elephants')
    assert isinstance(out, str)
    assert len(out.split()) >= 3  
    
def test_enc_decoder_generate_returns_text():
    tok = Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncoderDecoderTransformer().to(device)
    out = generate_joke(model, tok, 'Tell me a joke about elephants')
    assert isinstance(out, str)
    assert len(out.split()) >= 3  