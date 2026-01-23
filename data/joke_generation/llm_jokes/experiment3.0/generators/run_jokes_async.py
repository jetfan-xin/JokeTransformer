import asyncio
import csv
import os
import re
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)
import logging

'''
Reading input from /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/stats/top_5000_topic_combos.csv...
Output File: /ltstorage/home/4xin/uhh-ias-ml/data/llm_jokes/experiment3.0/outputs/deepseek_jokes.csv
Processing Range: [0:5000] (Total: 5000 combos)
Starting execution with Concurrency Limit = 50...
Generating Jokes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [3:10:34<00:00,  4.37it/s]
✅ Done. Success: 50000/50000
'''
# ==========================================
#               Core Configuration
# ==========================================

# Switch: True = test mode (run 3 combos, print results)
#         False = production mode (full speed, minimal logging)
TEST_MODE = False  

# Environment variable (export DEEPSEEK_API_KEY="sk-..." before running)
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

INPUT_CSV = BASE_DIR / "stats/top_5000_topic_combos.csv" 

# ==========================================
#           Auto Parameter Selection
# ==========================================
if TEST_MODE:
    print("\n" + "="*50)
    print("RUNNING IN TEST MODE (Dry Run) ")
    print("="*50 + "\n")
    OUTPUT_CSV = BASE_DIR / "outputs/deepseek_jokes_TEST.csv" # Test file; does not touch main data
    TARGET_JOKES_PER_COMBO = 5  # Test run generates 5
    BATCH_SIZE = 5              # Single request per batch
    MAX_CONCURRENCY = 2         # Low concurrency for easier logs
    START_INDEX = 567           
    END_INDEX = 570             # Run 3 combos (567, 568, 569)
    LOG_LEVEL = logging.INFO    
else:
    # --- Production settings ---
    OUTPUT_CSV = BASE_DIR / "outputs/deepseek_jokes.csv"
    TARGET_JOKES_PER_COMBO = 100
    BATCH_SIZE = 10         
    MAX_CONCURRENCY = 50    
    START_INDEX = 0         
    END_INDEX = 5000        # Run all 5000 combos
    LOG_LEVEL = logging.WARNING

# Logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ==========================================
#                 Helpers
# ==========================================

def clean_combo_text(text):
    return text.replace('"', '').strip()

def parse_numbered_list(text):
    lines = text.split('\n')
    jokes = []
    current_joke = []
    pattern = re.compile(r'^\d+[\.\)]\s*(.*)')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        match = pattern.match(line)
        if match:
            if current_joke: jokes.append(" ".join(current_joke))
            current_joke = [match.group(1)]
        else:
            if current_joke: current_joke.append(line)
    if current_joke:
        jokes.append(" ".join(current_joke))
    return jokes

def get_existing_progress(output_file):
    if not os.path.exists(output_file):
        return {}
    counts = {}
    with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            next(reader) 
        except StopIteration:
            return {}
        for row in reader:
            if not row: continue
            c = row[0]
            counts[c] = counts.get(c, 0) + 1
    return counts

# ==========================================
#              Core Async Task
# ==========================================

@retry(
    retry=retry_if_exception_type(Exception), 
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def generate_jokes_async(client, semaphore, combo, num_jokes_needed, file_lock, writer):
    async with semaphore:
        # Prompt optimized for natural integration
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
                temperature=1.3 
            )
            
            raw_text = response.choices[0].message.content.strip()
            generated_jokes = parse_numbered_list(raw_text)

            # Fallback if the model does not return a list
            if not generated_jokes and len(raw_text) > 10:
                generated_jokes = [raw_text]
            
            # Test mode: print output for inspection
            if TEST_MODE:
                print(f"\n[DEBUG] Combo: {combo}")
                for idx, joke in enumerate(generated_jokes, 1):
                    # Truncate long lines for readability
                    preview = joke[:80] + "..." if len(joke) > 80 else joke
                    print(f"  {idx}. {preview}")
                print("-" * 30)

            # Write output rows
            async with file_lock:
                for joke in generated_jokes:
                    writer.writerow([combo, prompt, joke, MODEL_NAME])
            
            return len(generated_jokes)

        except Exception as e:
            if "400" in str(e): return 0 
            raise e

async def main():
    if not API_KEY:
        raise ValueError("Error: DEEPSEEK_API_KEY environment variable not set.")

    # Initialize client
    client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    # 1. Read input
    print(f"Reading input from {INPUT_CSV}...")
    combos = []
    if os.path.exists(INPUT_CSV):
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header if present
            start_row = 1 if (lines and ("freq" in lines[0] or "combo" in lines[0])) else 0
            for line in lines[start_row:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    c = clean_combo_text(",".join(parts[:-1]))
                    if c: combos.append(c)
    else:
        print(f"File not found: {INPUT_CSV}")
        return

    # Slice range
    total_loaded = len(combos)
    final_end = END_INDEX if END_INDEX is not None else total_loaded
    # Clamp end index
    f_end = min(final_end, total_loaded)
    
    combos_slice = combos[START_INDEX:f_end]
    
    print(f"Output File: {OUTPUT_CSV}")
    print(f"Processing Range: [{START_INDEX}:{f_end}] (Total: {len(combos_slice)} combos)")
    
    if len(combos_slice) == 0:
        print("Warning: Combo list slice is empty! Check START_INDEX/END_INDEX.")
        return

    # 2. Check progress
    existing_counts = get_existing_progress(OUTPUT_CSV)
    
    # 3. Prepare output file
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)
    f_out = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8', buffering=1)
    writer = csv.writer(f_out)
    if not file_exists:
        writer.writerow(["combo", "batch_prompt", "joke_text", "model_version"])
    
    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # 4. Build tasks
    tasks = []
    for combo in combos_slice:
        current_count = existing_counts.get(combo, 0)
        
        # In test mode, ignore existing counts and run once
        if TEST_MODE:
            jokes_needed = TARGET_JOKES_PER_COMBO
        else:
            jokes_needed = TARGET_JOKES_PER_COMBO - current_count
        
        while jokes_needed > 0:
            this_batch = min(BATCH_SIZE, jokes_needed)
            tasks.append(generate_jokes_async(client, semaphore, combo, this_batch, file_lock, writer))
            jokes_needed -= this_batch

    if not tasks:
        print("No tasks to run (Target met).")
        f_out.close()
        return

    # 5. Run
    print(f"Starting execution with Concurrency Limit = {MAX_CONCURRENCY}...")
    # results = await tqdm.gather(*tasks, return_exceptions=True)

    # --- Manual progress bar for return_exceptions ---
    pbar = tqdm(total=len(tasks), desc="Generating Jokes")

    async def progress_wrapper(coro):
        try:
            return await coro
        finally:
            # Always update progress, success or failure
            pbar.update(1)

    # Wrap each task to update the progress bar
    wrapped_tasks = [progress_wrapper(t) for t in tasks]

    # Use asyncio.gather with return_exceptions
    results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    
    pbar.close()
    # --- End manual progress wrapper ---

    # 6. Summary
    success = sum(1 for r in results if isinstance(r, int))
    print(f"\nDone. Success: {success}/{len(tasks)}")
    f_out.close()

if __name__ == "__main__":
    asyncio.run(main())
