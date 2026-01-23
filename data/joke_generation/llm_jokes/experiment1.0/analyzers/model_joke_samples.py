import pandas as pd
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JOKES_FILE = BASE_DIR / "stats/merged_llm_jokes.csv"
OUTPUT_SAMPLES_JSON = BASE_DIR / "stats/model_joke_samples.json"

def clean_combo_text(text):
    """Clean extra quotes from combo text."""
    if pd.isna(text): return ""
    return str(text).replace('"', '').strip()

def main():
    print("Starting random joke sampling...")

    print(f"Reading jokes data: {JOKES_FILE}")
    try:
        df = pd.read_csv(JOKES_FILE, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df['clean_combo'] = df['combo'].apply(clean_combo_text)

    print("Sampling up to 50 jokes per model...")
    samples_by_model = {}
    
    if 'model_version' in df.columns:
        for model_name, group in df.groupby('model_version'):
            n_samples = min(50, len(group))
            sampled_rows = group.sample(n=n_samples, random_state=42)
            
            jokes_list = []
            for _, row in sampled_rows.iterrows():
                jokes_list.append({
                    "combo": row['clean_combo'],
                    "joke_text": row['joke_text'],
                })
            
            samples_by_model[str(model_name)] = jokes_list
            print(f"  - Model {model_name}: sampled {len(jokes_list)} jokes")
            
        print(f"Saving sampled jokes to: {OUTPUT_SAMPLES_JSON}")
        with open(OUTPUT_SAMPLES_JSON, 'w', encoding='utf-8') as f:
            json.dump(samples_by_model, f, indent=2, ensure_ascii=False)
        
        print("Sampling complete.")
    else:
        print("Warning: 'model_version' column not found; skipping sampling.")

if __name__ == "__main__":
    main()
