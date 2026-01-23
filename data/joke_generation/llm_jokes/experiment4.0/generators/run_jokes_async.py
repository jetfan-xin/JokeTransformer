import asyncio
import csv
import logging
import os
import re
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

"""
Example:
Output File: /.../outputs/deepseek_jokes_raw.csv
Target Jokes: 100000 (Existing: 0, Remaining: 100000)
Starting execution with Concurrency Limit = 50...
Generating Jokes: 100%|...| 100000/100000 [xx:xx<...]
Done. Generated: 100000/100000
"""
# ==========================================
#               Core Configuration
# ==========================================

# Switch: True = test mode (small run, print results)
#         False = production mode (full speed, minimal logging)
TEST_MODE = False

# Environment variable (export DEEPSEEK_API_KEY="sk-..." before running)
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# ==========================================
#           Auto Parameter Selection
# ==========================================
if TEST_MODE:
    print("\n" + "=" * 50)
    print("RUNNING IN TEST MODE (Dry Run)")
    print("=" * 50 + "\n")
    OUTPUT_CSV = BASE_DIR / "outputs/deepseek_jokes_raw_TEST.csv"
    TARGET_TOTAL_JOKES = 100
    JOKES_PER_REQUEST = 10
    MAX_CONCURRENCY = 2
    LOG_LEVEL = logging.INFO
else:
    OUTPUT_CSV = BASE_DIR / "outputs/deepseek_jokes_raw.csv"
    TARGET_TOTAL_JOKES = 100_000
    JOKES_PER_REQUEST = 5
    MAX_CONCURRENCY = 50
    LOG_LEVEL = logging.WARNING

# Logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ==========================================
#                 Helpers
# ==========================================


def parse_numbered_list(text):
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


def parse_joke_and_theme(raw_joke):
    raw_joke = raw_joke.strip()
    match = re.match(r"^(.*)\[([^\[\]]+)\]\s*$", raw_joke)
    if not match:
        return raw_joke, ""
    joke_text = match.group(1).strip()
    theme_text = match.group(2).strip()
    themes = [t.strip() for t in theme_text.split(",") if t.strip()]
    return joke_text, ",".join(themes)


def build_prompt(num_jokes):
    return (
        f"Please write {num_jokes} distinct, hilarious, and creative short jokes.\n"
        f"Output Format: Joke Content [Themes]\n"
        f"Theme Constraints: Tags must be limited to one or two words. "
        f"The list must contain at least one noun found verbatim in the joke text. "
        f"Themes must be separated by commas.\n\n"
        f"Strict Output Format: Output ONLY a numbered list from 1 to {num_jokes}. "
        f"No intro, no outro, no explanations. Just the jokes.\n\n"
        f"Format Example:\n"
        f"1. The first joke content [theme one, theme two]\n"
        f"2. The second joke content [theme]\n"
        f"... (and so on)\n\n"
        f"Your Turn:\n"
    )


def get_existing_count(output_file):
    if not os.path.exists(output_file):
        return 0
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if row)


# ==========================================
#              Core Async Task
# ==========================================


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def generate_jokes_async(client, semaphore, num_jokes_requested, file_lock, writer):
    async with semaphore:
        prompt = build_prompt(num_jokes_requested)
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

            # Fallback if the model does not return a list
            if not generated_jokes and len(raw_text) > 10:
                generated_jokes = [raw_text]

            generated_jokes = [j.strip() for j in generated_jokes if j.strip()]
            generated_jokes = generated_jokes[:num_jokes_requested]

            if TEST_MODE:
                print("\n[DEBUG] Batch Output")
                for idx, joke in enumerate(generated_jokes, 1):
                    preview = joke[:80] + "..." if len(joke) > 80 else joke
                    print(f"  {idx}. {preview}")
                print("-" * 30)

            async with file_lock:
                for raw_joke in generated_jokes:
                    joke_text, theme_text = parse_joke_and_theme(raw_joke)
                    writer.writerow([raw_joke, joke_text, theme_text])

            return len(generated_jokes)

        except Exception as e:
            if "400" in str(e):
                return 0
            raise e


async def main():
    if not API_KEY:
        raise ValueError("Error: DEEPSEEK_API_KEY environment variable not set.")

    # Initialize client
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    # Prepare output file
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)
    existing_count = get_existing_count(OUTPUT_CSV) if file_exists else 0
    if TEST_MODE:
        existing_count = 0

    remaining_total = TARGET_TOTAL_JOKES - existing_count
    print(f"Output File: {OUTPUT_CSV}")
    print(
        f"Target Jokes: {TARGET_TOTAL_JOKES} "
        f"(Existing: {existing_count}, Remaining: {remaining_total})"
    )
    if remaining_total <= 0:
        print("Target met. Nothing to generate.")
        return

    f_out = open(OUTPUT_CSV, "a", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(f_out)
    if not file_exists:
        writer.writerow(["raw_joke", "joke", "theme"])

    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    print(f"Starting execution with Concurrency Limit = {MAX_CONCURRENCY}...")
    pbar = tqdm(total=remaining_total, desc="Generating Jokes")

    produced_total = 0
    try:
        while produced_total < remaining_total:
            remaining = remaining_total - produced_total
            tasks = []
            for i in range(0, remaining, JOKES_PER_REQUEST):
                num_jokes = min(JOKES_PER_REQUEST, remaining - i)
                tasks.append(
                    generate_jokes_async(
                        client, semaphore, num_jokes, file_lock, writer
                    )
                )

            if not tasks:
                break

            async def progress_wrapper(coro):
                try:
                    result = await coro
                except Exception as exc:
                    logger.warning("Task failed after retries: %s", exc)
                    result = 0
                pbar.update(result)
                return result

            results = await asyncio.gather(*(progress_wrapper(t) for t in tasks))
            produced = sum(results)
            produced_total += produced

            if produced == 0:
                logger.warning(
                    "No jokes produced in this round; stopping to avoid infinite loop."
                )
                break
    finally:
        pbar.close()
        f_out.close()

    print(f"\nDone. Generated: {produced_total}/{remaining_total}")


if __name__ == "__main__":
    asyncio.run(main())
