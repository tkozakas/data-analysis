import os
import base64
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EMAIL_DIR = "emails"
MAX_PARALLEL = 10
MESSAGES_PER_BATCH = 20
TOTAL_TO_FETCH = 1000
MAX_RETRIES = 3

# Load credentials once
def load_creds():
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

def build_service(creds):
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

def fetch_and_save_blocking(creds, msg_id):
    # Build a dedicated service per call
    service = build_service(creds)
    msg_data = (
        service.users()
        .messages()
        .get(userId="me", id=msg_id, format="raw")
        .execute()
    )
    raw_msg = base64.urlsafe_b64decode(msg_data["raw"])
    os.makedirs(EMAIL_DIR, exist_ok=True)
    with open(f"{EMAIL_DIR}/{msg_id}.eml", "wb") as f:
        f.write(raw_msg)
    return msg_id

async def fetch_message(creds, msg_id, sem, executor):
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                loop = asyncio.get_running_loop()
                msg_id_returned = await loop.run_in_executor(
                    executor, fetch_and_save_blocking, creds, msg_id
                )
            print(f"Saved: {msg_id_returned}")
            return
        except HttpError as e:
            wait = (2 ** attempt) + random.random()
            print(f"Retry {attempt+1}/{MAX_RETRIES} for {msg_id}: {e}, wait {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"Error {attempt+1}/{MAX_RETRIES} for {msg_id}: {e}, wait {wait:.1f}s")
            await asyncio.sleep(wait)
    print(f"FAILED after {MAX_RETRIES} retries: {msg_id}")

async def fetch_all_messages(creds):
    # get one lightweight listing service only
    service = build_service(creds)
    all_ids = []
    next_page_token = None
    while len(all_ids) < TOTAL_TO_FETCH:
        result = (
            service.users()
            .messages()
            .list(userId="me", maxResults=MESSAGES_PER_BATCH, pageToken=next_page_token)
            .execute()
        )
        all_ids.extend([m["id"] for m in result.get("messages", [])])
        next_page_token = result.get("nextPageToken")
        if not next_page_token:
            break

    all_ids = all_ids[:TOTAL_TO_FETCH]
    print(f"Total messages to download: {len(all_ids)}")

    sem = asyncio.Semaphore(MAX_PARALLEL)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        for i in range(0, len(all_ids), 50):
            chunk = all_ids[i:i + 50]
            tasks = [fetch_message(creds, msg_id, sem, executor) for msg_id in chunk]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.2)

def main():
    creds = load_creds()
    asyncio.run(fetch_all_messages(creds))

if __name__ == "__main__":
    main()
