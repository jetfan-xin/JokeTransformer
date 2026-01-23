import os
import glob
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def merge_csv_files(input_folder, output_file):
    """
    Merge all CSV files in the given folder into a single file.

    Args:
        input_folder (str): Folder containing CSV files.
        output_file (str): Output path for the merged CSV.
    """
    search_path = os.path.join(input_folder, "*.csv")
    all_files = glob.glob(search_path)
    
    if not all_files:
        print(f"No CSV files found in folder '{input_folder}'.")
        return

    print(f"Found {len(all_files)} CSV files. Merging...")

    df_list = []
    
    for filename in all_files:
        try:
            try:
                df = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(filename, encoding='latin1', on_bad_lines='skip')
            
            if not df.empty:
                df_list.append(df)
                print(f"  Loaded: {os.path.basename(filename)} ({len(df)} rows)")
            else:
                print(f"  Skipping empty file: {os.path.basename(filename)}")
                
        except Exception as e:
            print(f"  Error reading file {filename}: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        
        initial_len = len(combined_df)
        combined_df.drop_duplicates(inplace=True)
        final_len = len(combined_df)
        if initial_len != final_len:
            print(f"  Removed {initial_len - final_len} duplicate rows")

        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print("\nMerge complete.")
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No valid data to merge.")

if __name__ == "__main__":
    INPUT_FOLDER = BASE_DIR / "raw_outputs"
    OUTPUT_FILE = BASE_DIR / "stats/merged_llm_jokes.csv"
    
    output_dir = OUTPUT_FILE.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_csv_files(INPUT_FOLDER, OUTPUT_FILE)
