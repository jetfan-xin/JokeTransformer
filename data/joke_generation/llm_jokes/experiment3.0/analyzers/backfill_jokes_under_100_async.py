import argparse
import asyncio
import csv
import math
import os
import re
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging


INPUT_CSV = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/llm_jokes_top_5000_topics_3_detox_safe.csv"
)
OUTPUT_CSV = Path(
    "/ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/backfilled_jokes.csv"
)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"

DEFAULT_TARGET = 100
DEFAULT_MULTIPLIER = 1.1
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_CONCURRENCY = 50


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_numbered_list(text: str):
    lines = text.split("\n")
    jokes = []
    current_joke = []
    pattern = re.compile(r"^\d+[\.\)]\s*(.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            if current_joke:
                jokes.append(" ".join(current_joke))
            current_joke = [match.group(1)]
        else:
            if current_joke:
                current_joke.append(line)
    if current_joke:
        jokes.append(" ".join(current_joke))
    return jokes


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def generate_jokes_async(client, semaphore, combo, num_jokes_needed, file_lock, writer):
    async with semaphore:
        prompt = (
            f"Task: Write exactly {num_jokes_needed} distinct, hilarious, and creative short jokes "
            f"that naturally include the following keywords/elements in every joke: [{combo}].\n\n"
            f"Strict Output Format: Output ONLY a numbered list from 1 to {num_jokes_needed}. "
            f"No intro, no outro, no explanations. Just the jokes.\n\n"
            f"Format Example:\n"
            f"1. [The first joke content goes here]\n"
            f"2. [The second joke content goes here]\n"
            f"... (and so on)\n\n"
            f"Your Turn (Keywords: [{combo}]):\n"
        )

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a witty comedian."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=1.3,
            )

            raw_text = response.choices[0].message.content.strip()
            generated_jokes = parse_numbered_list(raw_text)
            if not generated_jokes and len(raw_text) > 10:
                generated_jokes = [raw_text]

            async with file_lock:
                for joke in generated_jokes:
                    writer.writerow([combo, prompt, joke, MODEL_NAME])

            return len(generated_jokes)
        except Exception as e:
            if "400" in str(e):
                return 0
            raise e


def load_topic_counts(path: Path, topic_col: str) -> Counter:
    counts = Counter()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or topic_col not in reader.fieldnames:
            raise ValueError(f"Input CSV missing column '{topic_col}'.")
        for row in reader:
            topic = str(row.get(topic_col, "")).strip()
            if topic:
                counts[topic] += 1
    return counts


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    ap.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    ap.add_argument("--topic-col", default="topic")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--multiplier", type=float, default=DEFAULT_MULTIPLIER)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    ap.add_argument("--max-topics", type=int, help="Optional limit for debugging")
    args = ap.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if not API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    print(f"[LOAD] {args.input_csv}")
    counts = load_topic_counts(args.input_csv, args.topic_col)
    total_topics = len(counts)
    missing = {t: c for t, c in counts.items() if c < args.target}
    print(f"[INFO] topics: {total_topics}")
    print(f"[INFO] topics below {args.target}: {len(missing)}")

    if args.max_topics is not None:
        missing = dict(list(missing.items())[: args.max_topics])
        print(f"[INFO] limiting to first {len(missing)} topics for debug")

    total_to_generate = sum(
        math.ceil((args.target - c) * args.multiplier) for c in missing.values()
    )
    print(f"[PLAN] total jokes to generate: {total_to_generate}")

    if total_to_generate <= 0:
        print("[DONE] No topics need backfill.")
        return

    client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.output_csv.exists()
    f_out = open(args.output_csv, "a", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(f_out)
    if not file_exists:
        writer.writerow(["combo", "batch_prompt", "joke_text", "model_version"])

    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.max_concurrency)

    tasks = []
    for combo, count in missing.items():
        need = math.ceil((args.target - count) * args.multiplier)
        while need > 0:
            this_batch = min(args.batch_size, need)
            tasks.append(
                generate_jokes_async(client, semaphore, combo, this_batch, file_lock, writer)
            )
            need -= this_batch

    if not tasks:
        print("[DONE] No tasks to run.")
        f_out.close()
        return

    print(f"[RUN] tasks={len(tasks)} max_concurrency={args.max_concurrency}")
    pbar = tqdm(total=len(tasks), desc="Generating Jokes")

    async def progress_wrapper(coro):
        try:
            return await coro
        finally:
            pbar.update(1)

    wrapped_tasks = [progress_wrapper(t) for t in tasks]
    results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    pbar.close()

    success = sum(1 for r in results if isinstance(r, int))
    print(f"[DONE] Success: {success}/{len(tasks)}")
    f_out.close()


if __name__ == "__main__":
    asyncio.run(main())
