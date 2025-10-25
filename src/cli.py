import argparse
import json
from generate import generate
from eval import keyword_hit_rate
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--eval", action="store_true",
                    help="print JSON with text and score")
    args = ap.parse_args()

    text = generate(args.topic)
    score = keyword_hit_rate(text, args.topic)

    if args.eval:
        print(json.dumps({"topic": args.topic, "text": text, "keyword_hit_rate": score}, ensure_ascii=False))
    else:
        print(text)
        print(f"\n[metric] keyword_hit_rate: {score:.2f}")

if __name__ == "__main__":
    main()