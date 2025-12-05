import pandas as pd
import json
import re
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COMBOS_FILE = BASE_DIR / "stats/combos_top5000_across_k1_to_k5.csv"
JOKES_FILE = BASE_DIR / "stats/merged_llm_jokes.csv"
OUTPUT_CSV_SUMMARY = BASE_DIR / "stats/jokes_stats_summary.csv"

def clean_combo_text(text):
    """Clean extra quotes from combo text."""
    if pd.isna(text): return ""
    return str(text).replace('"', '').strip()

def main():
    print("Starting stats ordered by original index...")

    print(f"Reading baseline file: {COMBOS_FILE}")
    ordered_combos = []
    
    try:
        with open(COMBOS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_line = 0
            if lines and ("combo" in lines[0] or "freq" in lines[0]):
                start_line = 1
            
            for idx, line in enumerate(lines[start_line:]):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    raw_combo = ",".join(parts[:-1])
                    clean_c = clean_combo_text(raw_combo)
                    if clean_c:
                        ordered_combos.append({
                            "original_index": idx,
                            "combo": clean_c,
                            "total_generated": 0,
                            "details": {}
                        })
    except FileNotFoundError:
        print(f"Error: combos file not found: {COMBOS_FILE}")
        return

    print(f"Baseline list built with {len(ordered_combos)} combos.")

    print(f"Reading jokes data: {JOKES_FILE}")
    try:
        df = pd.read_csv(JOKES_FILE, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading jokes CSV: {e}")
        return

    df['clean_combo'] = df['combo'].apply(clean_combo_text)

    print("Aggregating stats per combo and model...")
    total_counts = df['clean_combo'].value_counts().to_dict()

    if 'model_version' in df.columns:
        model_counts = (
            df.groupby(['clean_combo', 'model_version']).size().to_dict()
        )
    else:
        model_counts = {}

    combo_to_list_index = {item['combo']: i for i, item in enumerate(ordered_combos)}
    for combo, count in total_counts.items():
        if combo in combo_to_list_index:
            idx = combo_to_list_index[combo]
            ordered_combos[idx]['total_generated'] = int(count)

    for (combo, model), count in model_counts.items():
        if combo in combo_to_list_index:
            idx = combo_to_list_index[combo]
            if model not in ordered_combos[idx]['details']:
                ordered_combos[idx]['details'][model] = 0
            ordered_combos[idx]['details'][model] += int(count)

    print(f"Saving CSV summary to: {OUTPUT_CSV_SUMMARY}")
    csv_data = []
    for item in ordered_combos:
        row = {
            "Index": item['original_index'],
            "Combo": item['combo'],
            "Total_Jokes": item['total_generated']
        }
        for model, count in item['details'].items():
            row[f"{model}"] = count
        csv_data.append(row)
        
    df_summary = pd.DataFrame(csv_data)
    df_summary = df_summary.fillna(0)
    if not df_summary.empty:
        cols = df_summary.columns.drop(['Combo'])
        for col in cols:
             if pd.api.types.is_numeric_dtype(df_summary[col]):
                df_summary[col] = df_summary[col].astype(int)
    
    df_summary.to_csv(OUTPUT_CSV_SUMMARY, index=False, encoding='utf-8')

    print("Done. Use the summary CSV to check which indices are still missing.")

    if not df_summary.empty:
        print("\n--- Preview of first 10 combos ---")
        print(df_summary.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
