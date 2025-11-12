import os
import torch

from models.decoder_only import DecoderOnlyTransformer
from utils.config import Config
from utils.inference import generate_joke


def main():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Build checkpoint path (assuming the script runs from project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(project_root, "best_decoder_only.pt")

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found at {ckpt_path}")
        print("Please train the model first (run: python train.py).")
        return

    # Load trained model
    model = DecoderOnlyTransformer(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # <-- important for inference

    # Interactive topic input
    while True:
        topics = input("\nEnter topics for your joke (or 'quit' to exit): ")
        if topics.strip().lower() in ["quit", "exit"]:
            break

        joke = generate_joke(model, topics)
        print("\nGenerated joke:\n", joke if joke else "[No joke generated]")


if __name__ == "__main__":
    main()