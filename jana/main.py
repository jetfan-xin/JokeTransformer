import argparse, torch
from data.dataset_loader import JokeDataset
from utils import train_utils, evaluation, inference, config, tokenizer
from models.decoder_only import DecoderOnlyTransformer
from models.encoder_decoder import EncoderDecoderTransformer
from torch.utils.data import DataLoader
from torch import nn, optim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--arch", choices=["decoder", "encdec"], required=True)
    parser.add_argument("--prompt", type=str, default="Tell me a joke about elephants")
    parser.add_argument("--eval", action='store_true', help="Enable evaluation mode")
    args = parser.parse_args()

    
    tok = tokenizer.Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer().to(device) if args.arch == "decoder" else EncoderDecoderTransformer().to(device)
    
    if args.mode == "train":
        print("Training the model.")

    elif args.mode == "infer":
        joke = inference.generate_joke(model, tok, args.prompt)
        print("🤖 Joke:", joke)

        if args.eval:
            score = evaluation.topic_inclusion(args.prompt, joke)
            print(f"🎯 Topic Inclusion Score: {score:.2f}")
            humour_score = evaluation.check_humour(joke)
            print(f"🤡 Humour or No Humour  : {humour_score['label']}")
            print(f"🎯 {humour_score['label']} Score         : {humour_score['score']:.2f}")
            

if __name__ == "__main__":
    main()