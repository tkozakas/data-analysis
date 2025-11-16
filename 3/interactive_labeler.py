import os
import shutil
import sys
import subprocess
import random
import time

EMAIL_DIR = "emails"
TARGET_COUNT = 200

CLOSE_WINDOW_ENABLED = False
EMAIL_CLIENT_PROCESS_NAME = "Mail"

GOOD_DIR = os.path.join(EMAIL_DIR, "good")
SPAM_DIR = os.path.join(EMAIL_DIR, "spam")

def open_file(filepath):
    try:
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.run(["open", filepath], check=True)
        else:
            subprocess.run(["xdg-open", filepath], check=True)
    except Exception as e:
        print(f"Error: Could not open file '{filepath}'.")
        print(f"Details: {e}")
        return False
    return True

def close_email_client():
    if not CLOSE_WINDOW_ENABLED or not EMAIL_CLIENT_PROCESS_NAME:
        return

    time.sleep(0.5)

    print(f"--> Attempting to close '{EMAIL_CLIENT_PROCESS_NAME}'...")
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/IM", EMAIL_CLIENT_PROCESS_NAME],
                check=False, capture_output=True
            )
        elif sys.platform in ["darwin", "linux"]:
            subprocess.run(
                ["killall", EMAIL_CLIENT_PROCESS_NAME],
                check=False, capture_output=True
            )
        print("--> Close command sent.")
    except FileNotFoundError:
        print(f"--> Error: Command not found. Could not execute kill command.")
    except Exception as e:
        print(f"--> An unexpected error occurred while trying to close the client: {e}")


def main():
    print("--- Interactive Email Labeler (with Auto-Close) ---")
    if CLOSE_WINDOW_ENABLED:
        print(f"!!! Auto-close is ENABLED for '{EMAIL_CLIENT_PROCESS_NAME}'. This will close the entire application. !!!")
    else:
        print("--- Auto-close is DISABLED. Edit the script to enable it. ---")
    print("  'g' -> Good | 's' -> Spam | 'k' -> Skip | 'q' -> Quit")
    print("-" * 30)

    os.makedirs(GOOD_DIR, exist_ok=True)
    os.makedirs(SPAM_DIR, exist_ok=True)

    files_to_label = [
        f for f in os.listdir(EMAIL_DIR)
        if f.endswith(".eml") and os.path.isfile(os.path.join(EMAIL_DIR, f))
    ]

    if not files_to_label:
        print("No .eml files found in the 'emails' directory to label. Exiting.")
        return

    random.shuffle(files_to_label)

    good_count = len(os.listdir(GOOD_DIR))
    spam_count = len(os.listdir(SPAM_DIR))
    labeled_count = good_count + spam_count
    
    initial_labeled_count = labeled_count
    print(f"You have already labeled {labeled_count} emails.")
    print(f"Goal is to reach {TARGET_COUNT} total labeled emails.")
    print("-" * 30)

    for filename in files_to_label:
        if labeled_count >= TARGET_COUNT:
            print(f"\nTarget of {TARGET_COUNT} labeled emails reached. Well done!")
            break

        file_path = os.path.join(EMAIL_DIR, filename)
        
        if not open_file(file_path):
            break
        
        time.sleep(1)

        while True:
            progress = f"[{labeled_count}/{TARGET_COUNT}]"
            prompt = f"{progress} File: '{filename}' -> Enter command: "
            choice = input(prompt).lower().strip()
            if choice in ['g', 's', 'k', 'q']:
                break
            else:
                print("Invalid command.")

        if choice == 'q':
            print("Quitting script.")
            break
        elif choice == 'k':
            print("--> Skipped.\n")
            close_email_client()
            continue
        elif choice == 'g':
            dest_path = os.path.join(GOOD_DIR, filename)
            shutil.move(file_path, dest_path)
            good_count += 1
            labeled_count += 1
            print(f"--> Moved to 'good'. (Good: {good_count}, Spam: {spam_count})\n")
            close_email_client()
        elif choice == 's':
            dest_path = os.path.join(SPAM_DIR, filename)
            shutil.move(file_path, dest_path)
            spam_count += 1
            labeled_count += 1
            print(f"--> Moved to 'spam'. (Good: {good_count}, Spam: {spam_count})\n")
            close_email_client()

    print("\n--- Labeling Complete ---")
    print(f"Total Labeled in Session: {labeled_count - initial_labeled_count}")
    print(f"Final Count -> Good: {good_count} | Spam: {spam_count}")
    print("You can now run 'generate_labels.py' to create the final JSONL file.")


if __name__ == "__main__":
    main()
