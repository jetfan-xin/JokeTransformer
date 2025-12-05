import csv
import torch
import os
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
OUTPUT_CSV = BASE_DIR / "raw_outputs/qwen3-30b_805-1000.csv"

START_INDEX = 805
END_INDEX = 1000

TARGET_JOKES_PER_COMBO = 100
BATCH_SIZE = 20

SKIP_THRESHOLD = 85

def clean_combo_text(text):
    """Remove double quotes and surrounding whitespace."""
    return text.replace('"', '').strip()

def parse_numbered_list(text):
    """Parse jokes from a numbered list in model output."""
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
    if current_joke: jokes.append(" ".join(current_joke))
    return jokes

def main():
    print(f"Reading input from {INPUT_CSV}...")
    all_combos = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        start_line = 1 if "freq" in lines[0] else 0
        for line in lines[start_line:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                raw_combo = ",".join(parts[:-1])
                combo_clean = clean_combo_text(raw_combo)
                if combo_clean:
                    all_combos.append(combo_clean)
    
    target_combos = all_combos[START_INDEX:END_INDEX]
    print(f"Processing {len(target_combos)} combos.")

    print(f"Loading model: {MODEL_NAME} with 4-bit NF4 quantization...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="cuda:0", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    existing_counts = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try:
                next(reader)
                for row in reader:
                    if row:
                        c = row[0]
                        existing_counts[c] = existing_counts.get(c, 0) + 1
            except StopIteration:
                pass
    
    f_out = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.writer(f_out)
    if os.stat(OUTPUT_CSV).st_size == 0:
        writer.writerow(["combo", "batch_prompt", "joke_text", "model_version"])

    for combo in tqdm(target_combos, desc="Generating (NF4 4-bit)"):
        current_count = existing_counts.get(combo, 0)
        
        if current_count >= TARGET_JOKES_PER_COMBO or current_count > SKIP_THRESHOLD:
            continue

        jokes_needed = TARGET_JOKES_PER_COMBO - current_count

        while jokes_needed > 0:
            this_batch_size = min(BATCH_SIZE, jokes_needed)
            
            user_prompt = (
                f"Write exactly {this_batch_size} distinct, hilarious, and creative short jokes "
                f"that naturally use the following concept/nouns: [{combo}].\n"
                f"Format: Output a numbered list from 1 to {this_batch_size}.\n"
                f"Constraints: No intro, no outro, no explanations. Just the jokes."
            )
            messages = [{"role": "system", "content": "You are a creative comedian AI."},
                        {"role": "user", "content": user_prompt}]
            
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repetition_penalty=1.05,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            parsed_jokes = parse_numbered_list(response_text)
            
            if not parsed_jokes and len(response_text.strip()) > 5:
                 parsed_jokes = [response_text.strip()]

            for joke in parsed_jokes:
                writer.writerow([combo, user_prompt, joke, "Qwen3-30B-A3B"])
            f_out.flush()
            
            batch_count = len(parsed_jokes)
            if batch_count == 0: break 
            
            jokes_needed -= batch_count
            current_count += batch_count
            
            if current_count > SKIP_THRESHOLD:
                break

    f_out.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
