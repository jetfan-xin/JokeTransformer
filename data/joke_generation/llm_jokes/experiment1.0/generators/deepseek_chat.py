import csv
import time
import os
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

API_KEY = "xxx"   # DeepSeek API key
MODEL_NAME = "deepseek-chat"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
OUTPUT_CSV = BASE_DIR / "raw_outputs/deepseek_jokes.csv"

TARGET_JOKES_PER_COMBO = 100
BATCH_SIZE = 10  # Number of jokes per API request

# Index range control
START_INDEX = 567       # inclusive
END_INDEX = 3000        # exclusive

def clean_combo_text(text):
    """Clean extra quotes from CSV, e.g. \"\"\"dad\"\"\" -> dad."""
    return text.replace('"', '').strip()

def setup_client():
    key = API_KEY
    if not key:
        raise ValueError("API Key not found. Please set API_KEY in config or DEEPSEEK_API_KEY env var.")
    
    # Initialize OpenAI client pointing to DeepSeek.
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")

def parse_numbered_list(text):
    """
    Parse a numbered list of jokes from model output.
    Expected format:
    1. Joke...
    2. Joke...
    """
    # Match lines starting with "1. ", "2. ", etc.
    lines = text.split('\n')
    jokes = []
    current_joke = []
    pattern = re.compile(r'^\d+[\.\)]\s*(.*)')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = pattern.match(line)
        if match:
            # Flush previously collected joke.
            if current_joke:
                jokes.append(" ".join(current_joke))
            current_joke = [match.group(1)]
        else:
            # Continuation of the previous joke.
            if current_joke:
                current_joke.append(line)
    
    # Add the last collected joke.
    if current_joke:
        jokes.append(" ".join(current_joke))
        
    return jokes

def get_existing_progress(output_file):
    """Scan output file and count how many jokes exist per combo."""
    if not os.path.exists(output_file):
        return {}
    
    counts = {}
    with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return {}
        # Assume CSV structure: combo, raw_prompt, joke_text.
        for row in reader:
            if not row: continue
            c = row[0]
            counts[c] = counts.get(c, 0) + 1
    return counts

def main():
    try:
        client = setup_client()
    except ValueError as e:
        print(e)
        return

    # 1. Read input CSV
    print(f"Reading input from {INPUT_CSV}...")
    combos = []
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_idx = 0
            if lines and ("freq" in lines[0] or "combo" in lines[0]):
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    raw_combo = ",".join(parts[:-1]) 
                    combo_clean = clean_combo_text(raw_combo)
                    if combo_clean:
                        combos.append(combo_clean)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    total_combos_loaded = len(combos)
    print(f"Loaded {total_combos_loaded} total combos.")

    # Handle index slicing
    final_end_index = END_INDEX if END_INDEX is not None else total_combos_loaded
    final_end_index = min(final_end_index, total_combos_loaded)
    current_start_index = max(0, START_INDEX)

    if current_start_index >= total_combos_loaded:
        print(f"Start index ({current_start_index}) is larger than total items ({total_combos_loaded}). Exiting.")
        return

    # Slice the combos list for this run
    combos_to_process = combos[current_start_index : final_end_index]
    print(f"Processing range: [{current_start_index} to {final_end_index}]")
    print(f"Total combos to process in this run: {len(combos_to_process)}")

    # 2. Load existing progress
    existing_counts = get_existing_progress(OUTPUT_CSV)
    print(f"Found existing progress for {len(existing_counts)} combos (total in file).")

    # 3. Prepare output file
    file_exists = os.path.exists(OUTPUT_CSV)
    f_out = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.writer(f_out)
    
    if not file_exists:
        writer.writerow(["combo", "batch_prompt", "joke_text", "model_version"])

    # 4. Main loop
    for i, combo in enumerate(tqdm(combos_to_process, desc=f"Processing [{current_start_index}-{final_end_index}]"), start=current_start_index):
        current_count = existing_counts.get(combo, 0)
        if current_count >= TARGET_JOKES_PER_COMBO:
            continue 
        jokes_needed = TARGET_JOKES_PER_COMBO - current_count
        # Inner loop: generate in batches
        while jokes_needed > 0:
            this_batch_size = min(BATCH_SIZE, jokes_needed)
            
            prompt = (
                f"Write exactly {this_batch_size} distinct, hilarious, and creative short jokes "
                f"that naturally use the following concept/nouns: [{combo}].\n"
                f"Format: Output a numbered list from 1 to {this_batch_size}.\n"
                f"Constraints: No intro, no outro, no explanations. Just the jokes."
            )
            try:
                # Call DeepSeek API.
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a creative comedian assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=1.3  # Increase creativity
                )
                
                raw_text = response.choices[0].message.content.strip()
                generated_jokes = parse_numbered_list(raw_text)
                
                # Fallback: if parsing fails but there is content, treat as one joke.
                if len(generated_jokes) == 0 and len(raw_text) > 10:
                    generated_jokes = [raw_text]

                for joke in generated_jokes:
                    writer.writerow([combo, prompt, joke, MODEL_NAME])
                f_out.flush()

                jokes_obtained = len(generated_jokes)
                if jokes_obtained == 0:
                    print(f"\nWarning: Got 0 jokes for {combo} (Index {i}). Retrying...")
                    time.sleep(1)
                    continue

                jokes_needed -= jokes_obtained
                
                # Light rate limiting between requests.
                time.sleep(0.2) 

            except Exception as e:
                error_str = str(e)
                print(f"\nError processing {combo} (Index {i}): {error_str[:100]}...")
                if "429" in error_str or "Rate limit" in error_str:
                    print("Hit rate limit. Sleeping for 30 seconds...")
                    time.sleep(5)
                elif "500" in error_str or "503" in error_str:
                    print("Server error, retrying in 5s...")
                    time.sleep(5)
                else:
                    print("Unknown error, skipping batch.")
                    break
    
    f_out.close()
    print("Done.")

if __name__ == "__main__":
    main()
