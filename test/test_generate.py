from src.generate import generate

def test_generate_returns_text():
    out = generate("elephants", max_tokens=6)
    assert isinstance(out, str)
    assert len(out.split()) >= 3  # something was generated