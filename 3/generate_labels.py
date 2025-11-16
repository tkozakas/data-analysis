import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
import random
from datetime import datetime, timedelta

from extract_features import parse_email, build_record, detect_time_segment

# --- Configuration ---
EMAIL_DIR = "emails"
OUTPUT_CSV = "email_features.csv"
NUM_SPAM_TO_USE = 200
# ---------------------

def generate_realistic_mock_metadata():
    """
    Creates randomized, realistic-looking metadata for mock emails.
    This now uses the imported detect_time_segment function.
    """
    SPAM_DOMAINS = [
        "xyz", "top", "club", "info", "online", "buzz", "site", "live",
        "click", "link", "shop", "loan", "win", "download",
        "promo-source.com", "marketing-deals.net", "special-offer.org",
        "daily-newsletter.com", "your-rewards.info"
    ]
    SENDER_NAMES = ["promo", "newsletter", "support", "contact", "offers", "deals", "admin"]
    
    domain = f"{random.choice(SENDER_NAMES)}-{random.randint(10,99)}.{random.choice(SPAM_DOMAINS)}"
    sender_local_part = random.choice(SENDER_NAMES)
    from_address = f"{sender_local_part}@{domain}"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    random_date = start_date + (end_date - start_date) * random.random()
    
    day_of_week = random_date.strftime("%A")
    time_of_day = detect_time_segment(random_date)

    return {
        "from": from_address,
        "day_of_week": day_of_week,
        "time_of_day": time_of_day
    }


def process_user_emails():
    """Processes the .eml files that the user has manually sorted."""
    print("--- Processing user's manually labeled emails ---")
    user_records = []
    good_dir = os.path.join(EMAIL_DIR, "good")
    spam_dir = os.path.join(EMAIL_DIR, "spam")

    if os.path.exists(good_dir):
        good_files = [f for f in os.listdir(good_dir) if f.endswith(".eml")]
        print(f"Found {len(good_files)} emails in 'emails/good/'")
        for filename in tqdm(good_files, desc="Processing good emails"):
            path = os.path.join(good_dir, filename)
            msg_data = parse_email(path)
            record = build_record(msg_data, "Good")
            user_records.append(record)

    if os.path.exists(spam_dir):
        spam_files = [f for f in os.listdir(spam_dir) if f.endswith(".eml")]
        print(f"Found {len(spam_files)} emails in 'emails/spam/'")
        for filename in tqdm(spam_files, desc="Processing spam emails"):
            path = os.path.join(spam_dir, filename)
            msg_data = parse_email(path)
            record = build_record(msg_data, "Spam")
            user_records.append(record)
    return user_records

def process_huggingface_spam():
    """
    Downloads a spam dataset from Hugging Face and runs feature extraction
    using realistic, randomized mock metadata.
    """
    print("\n--- Processing Hugging Face spam dataset ---")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    # Corrected variable name from h_token to hf_token
    if not hf_token:
        print("\nFATAL ERROR: Hugging Face token not found.")
        print("Please create a '.env' file with the line: HF_TOKEN='your_token_here'")
        return []
        
    try:
        print("Downloading dataset 'Deysi/spam-detection-dataset'...")
        ds = load_dataset("Deysi/spam-detection-dataset", token=hf_token)
        df = ds['train'].to_pandas()

    except Exception as e:
        print("\nFATAL ERROR: Failed to load data from Hugging Face.")
        print(f"Details: {e}")
        return []

    print("Successfully loaded data from Hugging Face.")
    spam_df = df[df['label'] == 'spam'].head(NUM_SPAM_TO_USE)
    
    if len(spam_df) < NUM_SPAM_TO_USE:
        print(f"Warning: Found only {len(spam_df)} spam emails, less than the requested {NUM_SPAM_TO_USE}.")

    print(f"Processing {len(spam_df)} spam emails from the dataset...")
    hf_records = []
    for index, row in tqdm(spam_df.iterrows(), total=len(spam_df), desc="Processing Hugging Face spam"):
        full_text = str(row['text'])
        subject = ""
        body = full_text
        if full_text.startswith("Subject:"):
            parts = full_text.split('\n', 1)
            subject = parts[0].replace("Subject:", "").strip()
            if len(parts) > 1:
                body = parts[1].strip()
        
        mock_metadata = generate_realistic_mock_metadata()
        
        mock_msg_data = {
            "id": f"hf_{index}",
            "from": mock_metadata["from"],
            "cc": "",
            "subject": subject,
            "body": body,
            "day_of_week": mock_metadata["day_of_week"],
            "time_of_day": mock_metadata["time_of_day"]
        }
        record = build_record(mock_msg_data, "Spam")
        hf_records.append(record)
    return hf_records

def main():
    user_email_records = process_user_emails()
    hf_spam_records = process_huggingface_spam()
    all_records = user_email_records + hf_spam_records
    
    if not all_records:
        print("\nNo records were generated. Exiting.")
        return
        
    print(f"\n--- Total records generated: {len(all_records)} ---")
    print(f"User records: {len(user_email_records)}")
    print(f"Hugging Face records: {len(hf_spam_records)}")

    final_df = pd.DataFrame(all_records)
    if 'label' in final_df.columns:
        cols = [c for c in final_df.columns if c != 'label'] + ['label']
        final_df = final_df[cols]

    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nSuccessfully created final dataset at '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
