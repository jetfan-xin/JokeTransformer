import pandas as pd
import json
import re
import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COMBOS_FILE = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
JOKES_FILE = BASE_DIR / "stats/merged_llm_jokes.csv"
OUTPUT_JSON = BASE_DIR / "stats/llm_jokes_stats.json"

def clean_combo_text(text):
    """
    Clean combo text by removing extra quotes from CSV.
    """
    if pd.isna(text):
        return ""
    return str(text).replace('"', '').strip()

def extract_batch_size(prompt):
    """
    Extract the requested batch size from the prompt, e.g. "Write exactly 20 distinct...".
    """
    if pd.isna(prompt):
        return "unknown"
    
    match = re.search(r"Write exactly (\d+)", str(prompt))
    if match:
        return int(match.group(1))
    return "unknown"

def main():
    print("Starting analysis...")

    print(f"Reading combo index file: {COMBOS_FILE}")
    combo_index_map = {}
    
    try:
        with open(COMBOS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_line = 0
            if "combo" in lines[0] and "freq" in lines[0]:
                start_line = 1 
            
            data_row_index = 0
            for line in lines[start_line:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    raw_combo = ",".join(parts[:-1])
                    clean_c = clean_combo_text(raw_combo)
                    if clean_c:
                        combo_index_map[clean_c] = data_row_index
                        data_row_index += 1
                        
        print(f"Loaded {len(combo_index_map)} combos into index map.")
        
    except FileNotFoundError:
        print(f"Error: combo index file not found: {COMBOS_FILE}")
        return

    print(f"Reading jokes dataset: {JOKES_FILE}")
    try:
        df = pd.read_csv(JOKES_FILE, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading jokes CSV: {e}")
        return

    print(f"Dataset has {len(df)} rows. Processing...")

    df['clean_combo'] = df['combo'].apply(clean_combo_text)
    df['requested_size'] = df['batch_prompt'].apply(extract_batch_size)

    model_global_counts = df['model_version'].value_counts().to_dict()
    stats_structure = {}

    grouped = df.groupby(['model_version', 'requested_size'])

    for (model, size), group_df in grouped:
        model = str(model)
        size_key = f"batch_size_{size}"
        
        if model not in stats_structure:
            stats_structure[model] = {
                "total_model_count": int(model_global_counts.get(model, 0))
            }
        
        combo_counts = group_df['clean_combo'].value_counts().to_dict()
        
        combos_detail = {}
        for combo_name, count in combo_counts.items():
            original_idx = combo_index_map.get(combo_name, -1)
            combos_detail[combo_name] = {
                "original_index": original_idx,
                "generated_count": count
            }

        stats_structure[model][size_key] = {
            "total_jokes_generated": int(len(group_df)),
            "unique_combos_count": len(combos_detail),
            "combos_detail": combos_detail
        }

    print(f"Saving stats to {OUTPUT_JSON}...")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats_structure, f, indent=4, ensure_ascii=False)

    print("Analysis complete.")
    
    for model in stats_structure:
        total = stats_structure[model].get("total_model_count", 0)
        print(f"\nModel: {model} (total: {total} rows)")
        
        for key in stats_structure[model]:
            if key == "total_model_count":
                continue
            
            sub_count = stats_structure[model][key]["total_jokes_generated"]
            print(f"  - {key}: {sub_count}")

if __name__ == "__main__":
    main()
