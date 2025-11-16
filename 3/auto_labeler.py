import os
import shutil
import random
from email.parser import BytesParser
from email import policy
from tqdm import tqdm

# --- Configuration ---
EMAIL_DIR = "emails"
TARGET_COUNT = 350  # The desired number of emails in each final folder
TRANSACTIONAL_KEYWORDS = [
    # English Keywords
    "order confirmation", "your receipt", "your invoice", "has shipped",
    "shipping confirmation", "tracking number", "order #", "invoice #",
    "e-receipt", "payment confirmation", "order has been placed", "your order is complete",
    "thank you for your order", "your purchase", "order details",
    "welcome to", "confirm your email", "verify your account", "password reset",
    "your account has been updated", "security alert", "new sign-in", "login attempt",
    "your verification code", "one-time password", "email address has been changed",
    "your statement", "your bill is ready", "payment due", "automatic payment",
    "billing statement", "subscription confirmation", "payment received",
    "your subscription is expiring", "account summary",
    "booking confirmation", "your ticket", "your flight is confirmed", "e-ticket",
    "your itinerary", "reservation details", "your booking", "rental confirmation",
    "your report is ready", "export is complete", "download is ready",
    "appointment confirmed", "your meeting is scheduled",
    # Lithuanian Keywords
    "užsakymo patvirtinimas", "sąskaita faktūra", "jūsų kvitas", "buvo išsiųstas",
    "siuntos sekimas", "užsakymas #", "sąskaita #", "elektroninis kvitas",
    "mokėjimo patvirtinimas", "užsakymas priimtas", "jūsų užsakymas įvykdytas",
    "dėkojame už užsakymą", "jūsų pirkimas", "užsakymo detalės",
    "sveiki atvykę", "patvirtinkite savo el. paštą", "patvirtinkite paskyrą",
    "slaptažodžio atstatymas", "jūsų paskyra atnaujinta", "saugumo pranešimas",
    "naujas prisijungimas", "bandymas prisijungti", "jūsų patvirtinimo kodas",
    "vienkartinis slaptažodis", "el. pašto adresas buvo pakeistas",
    "jūsų ataskaita", "jūsų sąskaita paruošta", "mokėjimo terminas",
    "automatinis mokėjimas", "atsiskaitymo išrašas", "prenumeratos patvirtinimas",
    "mokėjimas gautas", "jūsų prenumerata baigia galioti", "paskyros suvestinė",
    "rezervacijos patvirtinimas", "jūsų bilietas", "jūsų skrydis patvirtintas", "e. bilietas",
    "jūsų maršrutas", "rezervacijos detalės", "jūsų rezervacija", "nuomos patvirtinimas",
    "jūsų ataskaita paruošta", "eksportavimas baigtas", "atsisiuntimas paruoštas",
    "vizitas patvirtintas", "jūsų susitikimas suplanuotas"
]
# ---------------------

def get_email_content(filepath):
    """Parses an .eml file and returns its content as a single string."""
    try:
        with open(filepath, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = msg['subject'] or ""
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ("text/plain", "text/html"):
                    try: body_parts.append(part.get_content())
                    except Exception:
                        payload = part.get_payload(decode=True)
                        if payload: body_parts.append(payload.decode("utf-8", "ignore"))
        else:
            try: body_parts.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload: body_parts.append(payload.decode("utf-8", "ignore"))
        return (subject + " " + "\n".join(body_parts)).lower()
    except Exception as e:
        print(f"Warning: Could not read {filepath}. Error: {e}")
        return ""

def balance_directory(dir_path, target_count):
    """Randomly deletes files from a directory to match the target count."""
    print(f"\n--- Balancing directory: {dir_path} ---")
    if not os.path.isdir(dir_path):
        print("Directory not found. Skipping.")
        return

    files = [f for f in os.listdir(dir_path) if f.endswith(".eml")]
    current_count = len(files)
    print(f"Found {current_count} files.")

    num_to_delete = current_count - target_count
    
    if num_to_delete > 0:
        print(f"Need to delete {num_to_delete} random files to reach target of {target_count}.")
        
        # Get a random sample of files to delete
        files_to_delete = random.sample(files, num_to_delete)
        
        for filename in tqdm(files_to_delete, desc="Deleting excess files"):
            filepath_to_delete = os.path.join(dir_path, filename)
            os.remove(filepath_to_delete)
        
        print("Deletion complete.")
    else:
        print("File count is at or below the target. No files deleted.")

def main():
    print("--- Automatic Email Sorter & Balancer ---")
    
    transactional_dir = os.path.join(EMAIL_DIR, "transactional")
    other_dir = os.path.join(EMAIL_DIR, "other")

    # Clean up old directories before starting
    if os.path.exists(transactional_dir): shutil.rmtree(transactional_dir)
    if os.path.exists(other_dir): shutil.rmtree(other_dir)
    os.makedirs(transactional_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    
    files_to_sort = [
        f for f in os.listdir(EMAIL_DIR)
        if f.endswith(".eml") and os.path.isfile(os.path.join(EMAIL_DIR, f))
    ]

    if not files_to_sort:
        print("No .eml files found in the root 'emails' directory to sort. Exiting.")
        return

    # --- PHASE 1: SORTING (COPYING) ---
    print(f"\n--- Phase 1: Sorting {len(files_to_sort)} emails into categories ---")
    for filename in tqdm(files_to_sort, desc="Sorting emails"):
        filepath = os.path.join(EMAIL_DIR, filename)
        content = get_email_content(filepath)
        is_transactional = any(keyword in content for keyword in TRANSACTIONAL_KEYWORDS)
        
        if is_transactional:
            shutil.copy(filepath, os.path.join(transactional_dir, filename))
        else:
            shutil.copy(filepath, os.path.join(other_dir, filename))

    print("\n--- Sorting Phase Complete ---")
    
    # --- PHASE 2: BALANCING (DELETING FROM COPIES) ---
    print(f"\n--- Phase 2: Balancing folders to target count of {TARGET_COUNT} ---")
    balance_directory(transactional_dir, TARGET_COUNT)
    balance_directory(other_dir, TARGET_COUNT)
    
    # --- FINAL VERIFICATION ---
    print("\n--- Final Verification ---")
    final_t_count = len([f for f in os.listdir(transactional_dir) if f.endswith(".eml")])
    final_o_count = len([f for f in os.listdir(other_dir) if f.endswith(".eml")])
    print(f"Final count in '{transactional_dir}': {final_t_count} files.")
    print(f"Final count in '{other_dir}': {final_o_count} files.")
    print("\nProcess complete. You can now run build_dataset.py.")

if __name__ == "__main__":
    main()
