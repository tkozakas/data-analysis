import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from extract_features import parse_email, build_record

EMAIL_DIR = "emails"
OUTPUT_CSV_PATH = Path("data/email_features.csv")

def process_user_emails():
    print("--- Processing automatically labeled emails ---")
    all_records = []
    
    label_map = {
        "transactional": "Transactional",
        "other": "Other"
    }

    for folder_name, label in label_map.items():
        folder_path = os.path.join(EMAIL_DIR, folder_name)
        if not os.path.isdir(folder_path): continue

        files_in_folder = [f for f in os.listdir(folder_path) if f.endswith(".eml")]
        if not files_in_folder: continue
            
        print(f"Found {len(files_in_folder)} emails in '{folder_path}'")
        for filename in tqdm(files_in_folder, desc=f"Processing {folder_name} emails"):
            path = os.path.join(folder_path, filename)
            msg_data = parse_email(path)
            record = build_record(msg_data, label)
            all_records.append(record)
            
    return all_records

def main():
    all_records = process_user_emails()
    
    if not all_records:
        print("\nNo records were generated. Run auto_labeler.py first. Exiting.")
        return
    
    final_df = pd.DataFrame(all_records)

    print("\n--- Pre-filtering data to be Weka-friendly ---")
    
    for col in final_df.columns:
        if final_df[col].dtype == 'object' and col != 'label':
            final_df[col] = pd.factorize(final_df[col])[0]
        elif final_df[col].dtype == 'bool':
            final_df[col] = final_df[col].astype(int)

    if 'label' in final_df.columns:
        label_to_numeric_map = {'Other': 0, 'Transactional': 1}
        final_df['label'] = final_df['label'].map(label_to_numeric_map)
        print(f"Converted 'label' to numeric codes: {label_to_numeric_map}")
    
    if 'label' in final_df.columns:
        cols = [c for c in final_df.columns if c != 'label'] + ['label']
        final_df = final_df[cols]
    
    print(f"\nFinal dataset has {len(final_df)} records.")
    if 'label' in final_df.columns and not final_df['label'].dropna().empty:
        print("Value counts in final DataFrame:\n", final_df['label'].value_counts())
    
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"\nSuccessfully created Weka-friendly numeric CSV at '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()
