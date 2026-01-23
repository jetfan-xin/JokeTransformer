import csv
import time
import os
import re
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold

API_KEY = "xxx"
MODEL_NAME = "gemini-2.5-pro"  # e.g. gemini-2.5-flash-lite, gemini-2.5-flash

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
OUTPUT_CSV = BASE_DIR / "raw_outputs/gemini_jokes.csv"

TARGET_JOKES_PER_COMBO = 100
BATCH_SIZE = 50

START_INDEX = 567
END_INDEX = 3000
def clean_combo_text(text):
    """Clean extra quotes from CSV, e.g. \"\"\"dad\"\"\" -> dad."""
    return text.replace('"', '').strip()
def setup_model():
    """Configure and return a Gemini GenerativeModel."""
    key = API_KEY if API_KEY else os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("API key not found. Set API_KEY in config or GOOGLE_API_KEY env var.")
        
    genai.configure(api_key=key)
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_output_tokens=65535,
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

def parse_numbered_list(text):
    """
    Parse a numbered list of jokes from model output.

    Expected format:
    1. Joke...
    2. Joke...
    """
    lines = text.split('\n')
    jokes = []
    current_joke = []
    
    pattern = re.compile(r'^\d+[\.\)]\s*(.*)')
    
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

def get_existing_progress(output_file):
    """Scan output file and count jokes per combo."""
    if not os.path.exists(output_file):
        return {}
    
    counts = {}
    with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return {}
            
        for row in reader:
            if not row: continue
            c = row[0]
            counts[c] = counts.get(c, 0) + 1
    return counts

def main():
    try:
        model = setup_model()
    except ValueError as e:
        print(e)
        return

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

    final_end_index = END_INDEX if END_INDEX is not None else total_combos_loaded
    final_end_index = min(final_end_index, total_combos_loaded)
    current_start_index = max(0, START_INDEX)

    if current_start_index >= total_combos_loaded:
        print(f"Start index ({current_start_index}) is larger than total items ({total_combos_loaded}). Exiting.")
        return

    combos_to_process = combos[current_start_index : final_end_index]
    print(f"Processing range: [{current_start_index} to {final_end_index}]")
    print(f"Total combos to process in this run: {len(combos_to_process)}")

    existing_counts = get_existing_progress(OUTPUT_CSV)
    print(f"Found existing progress for {len(existing_counts)} combos (total in file).")

    file_exists = os.path.exists(OUTPUT_CSV)
    f_out = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.writer(f_out)
    
    if not file_exists:
        writer.writerow(["combo", "batch_prompt", "joke_text", "model_version"])

    for i, combo in enumerate(tqdm(combos_to_process, desc=f"Processing [{current_start_index}-{final_end_index}]"), start=current_start_index):
        
        current_count = existing_counts.get(combo, 0)
        
        if current_count >= TARGET_JOKES_PER_COMBO:
            continue

        jokes_needed = TARGET_JOKES_PER_COMBO - current_count
        
        while jokes_needed > 0:
            this_batch_size = min(BATCH_SIZE, jokes_needed)
            
            prompt = (
                f"Write exactly {this_batch_size} distinct, hilarious, and creative short jokes "
                f"that naturally use the following concept/nouns: [{combo}].\n"
                f"Format: Output a numbered list from 1 to {this_batch_size}.\n"
                f"Constraints: No intro, no outro, no explanations. Just the jokes."
            )

            try:
                response = model.generate_content(prompt)
                
                raw_text = response.text.strip()
                generated_jokes = parse_numbered_list(raw_text)
                
                if len(generated_jokes) == 0 and len(raw_text) > 10:
                    generated_jokes = [raw_text]

                for joke in generated_jokes:
                    writer.writerow([combo, prompt, joke, MODEL_NAME])
                
                f_out.flush()

                jokes_obtained = len(generated_jokes)
                if jokes_obtained == 0:
                    print(f"\nWarning: Got 0 jokes for {combo} (Index {i}). Retrying...")
                    time.sleep(0.5)
                    continue

                jokes_needed -= jokes_obtained
                
                time.sleep(0.5) 

            except Exception as e:
                print(f"\nError processing {combo} (Index {i}): {e}")
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    print("Hit rate limit. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print("Skipping batch due to error.")
                    break
    
    f_out.close()
    print("Done.")

if __name__ == "__main__":
    main()
