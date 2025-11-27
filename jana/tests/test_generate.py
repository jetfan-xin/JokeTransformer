from pathlib import Path
import torch
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from eval.run_eval import load_model_and_tokenizer, generate_joke, format_prompt_from_topics


def main():
    model_ckpt = Path("../models/checkpoints/checkpoint_49500.pt")  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading model from {model_ckpt} on {device}...")
    model_bundle = load_model_and_tokenizer(model_ckpt, device=device)

    # --- 2) Define some test topics ---
    examples = [
        ["dog", "cat"],
        ["elephant", "friend"],
        ["dentist", "tooth"],
        ["man", "woman", "bar"],
    ]

    for topics in examples:
        prompt = format_prompt_from_topics(topics)
        print("\n============================================================")
        print(f"Topics   : {topics}")
        print(f"Prompt   : {prompt!r}")

        for temp, top_k in [(1.0, 0), (0.8, 50)]:
            print(f"\n[GEN] temperature={temp}, top_k={top_k}")
            joke = generate_joke(
                model_bundle=model_bundle,
                prompt=prompt,
                max_new_tokens=64,
                temperature=temp,
                top_k=top_k,
            )
            print(joke)


if __name__ == "__main__":
    main()
    