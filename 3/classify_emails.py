import os
import re
import json
import csv
import string
import datetime
import emoji
import email
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

EMAIL_DIR = "emails"
LABELED_JSONL = "emails_labeled.jsonl"
OUTPUT_CSV = "email_features.csv"


# ---------- Feature helper functions ---------- #
def extract_domain(addr):
    m = re.search(r"@([A-Za-z0-9.-]+)", addr or "")
    return m.group(1).lower() if m else ""


def extract_first_email(addr):
    m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+)", addr or "")
    return m.group(1).lower() if m else ""


def get_recipient_count(to_field, cc_field):
    text = (to_field or "") + "," + (cc_field or "")
    return len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", text))


def detect_time_segment(dt: datetime.datetime):
    h = dt.hour
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 22:
        return "Evening"
    return "Night"


# ---------- Feature extractors registry ---------- #
def feature_from_domain(msg):
    return extract_domain(msg.get("from", ""))


def feature_recipient(msg):
    return extract_first_email(msg.get("to", ""))


def feature_body_word_count(msg):
    return len(re.findall(r"\w+", msg["body"]))


def feature_subject_word_count(msg):
    return len(re.findall(r"\w+", msg["subject"] or ""))


def feature_capitals(msg):
    txt = msg["body"] + " " + (msg["subject"] or "")
    return sum(c.isupper() for c in txt)


def feature_letters(msg):
    txt = msg["body"] + " " + (msg["subject"] or "")
    return sum(c.isalpha() for c in txt)


def feature_capital_ratio(msg):
    cap = feature_capitals(msg)
    letters = feature_letters(msg)
    return round(cap / letters, 4) if letters else 0


def feature_links(msg):
    return len(re.findall(r"https?://\S+", msg["body"]))


def feature_punctuation_count(msg):
    return sum(ch in string.punctuation for ch in msg["body"])


def feature_punctuation_ratio(msg):
    punct = feature_punctuation_count(msg)
    length = len(msg["body"])
    return round(punct / length, 4) if length else 0


def feature_emojis(msg):
    txt = msg["subject"] + " " + msg["body"]
    return sum(1 for ch in txt if emoji.is_emoji(ch))


def feature_emoji_ratio(msg):
    count = feature_emojis(msg)
    length = len(msg["body"])
    return round(count / length, 4) if length else 0


def feature_day_of_week(msg):
    return msg.get("day_of_week", "")


def feature_time_of_day(msg):
    return msg.get("time_of_day", "")


def feature_recipient_count(msg):
    return get_recipient_count(msg.get("to"), msg.get("cc"))


def feature_exclamation_count(msg):
    return msg["body"].count("!")


def feature_has_unsubscribe(msg):
    return bool(re.search(r"unsubscribe", msg["body"], re.IGNORECASE))


def feature_has_important(msg):
    txt = msg["subject"] + " " + msg["body"]
    return bool(re.search(r"\bimportant\b", txt, re.IGNORECASE))


# Mapping of feature name → function
FEATURES = {
    "from_domain": feature_from_domain,
    "recipient": feature_recipient,
    "recipient_count": feature_recipient_count,
    "body_word_count": feature_body_word_count,
    "subject_word_count": feature_subject_word_count,
    "capital_letter_count": feature_capitals,
    "capital_letter_ratio": feature_capital_ratio,
    "link_count": feature_links,
    "punctuation_count": feature_punctuation_count,
    "punctuation_ratio": feature_punctuation_ratio,
    "emoji_count": feature_emojis,
    "emoji_ratio": feature_emoji_ratio,
    "letter_count": feature_letters,
    "day_of_week": feature_day_of_week,
    "time_of_day": feature_time_of_day,
    "exclamation_mark_count": feature_exclamation_count,
    "has_unsubscribe": feature_has_unsubscribe,
    "has_important": feature_has_important,
}


# ---------- Email parsing ---------- #
def parse_email(path):
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    def get_header(name):
        return msg[name] or ""

    # Extract date/time info
    date_raw = get_header("Date")
    try:
        dt = parsedate_to_datetime(date_raw)
        day_name = dt.strftime("%A")
        time_seg = detect_time_segment(dt)
    except Exception:
        day_name, time_seg = "", ""

    from_ = get_header("From")
    to = get_header("To")
    cc = get_header("Cc")
    subject = get_header("Subject")

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in ("text/plain", "text/html"):
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode("utf-8", "ignore"))
    else:
        try:
            body_parts.append(msg.get_content())
        except Exception:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", "ignore"))

    return {
        "id": os.path.splitext(os.path.basename(path))[0],
        "from": from_,
        "to": to,
        "cc": cc,
        "subject": subject or "",
        "body": "\n".join(body_parts),
        "day_of_week": day_name,
        "time_of_day": time_seg,
    }


# ---------- Main pipeline ---------- #
def load_labels():
    labels = {}
    with open(LABELED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            labels[d["id"]] = d.get("label")
    return labels


def build_record(msg, label):
    record = {"id": msg["id"], "label": label}
    for name, func in FEATURES.items():
        try:
            record[name] = func(msg)
        except Exception:
            record[name] = None
    return record


def main():
    labels = load_labels()
    files = [f for f in os.listdir(EMAIL_DIR) if f.endswith(".eml")]
    print(f"Found {len(files)} emails, {len(labels)} labeled entries.")

    records = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(EMAIL_DIR, fname)
        msg = parse_email(path)
        label = labels.get(msg["id"])
        if not label:
            continue
        record = build_record(msg, label)
        records.append(record)
        if i % 50 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}")

    if not records:
        print("No records generated.")
        return

    fieldnames = list(records[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()