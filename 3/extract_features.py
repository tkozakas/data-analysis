# extract_features.py (Curated to 25 features)
import csv
import datetime
import os
import re
import string
import unicodedata
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime

import emoji

EMAIL_DIR = "emails"
LABELED_JSONL = "emails_labeled.jsonl"
OUTPUT_CSV = "email_features.csv"


# ---------- Feature helper functions ---------- #
def extract_domain(addr):
    m = re.search(r"@([A-Za-z0-9.-]+)", addr or "")
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


def feature_has_important(msg):
    txt = msg["subject"] + " " + msg["body"]
    return bool(re.search(r"\bimportant\b", txt, re.IGNORECASE))


def feature_longest_sentence_length(msg):
    text = msg["body"]
    sentences = re.split(r"[.!?]+", text)
    longest = 0
    for s in sentences:
        s_clean = re.sub(r"[^\w\d" + re.escape(string.punctuation) + r"\s]", "", s)
        words = re.findall(r"\w+", s_clean)
        longest = max(longest, len(words))
    return longest


def feature_question_mark_count(msg):
    return msg["body"].count("?")


def feature_feedback_words(msg):
    text = msg["body"]
    text_norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text_low = text_norm.lower()
    keywords = [
        "feedback", "recommendation", "reference", "review", "opinion",
        "thoughts", "assessment", "evaluation", "input", "comments",
        "insights", "testimonial", "atsiliepimas", "rekomendacija",
        "apžvalga", "nuomonė", "mintys", "įvertinimas", "vertinimas",
        "indėlis", "komentarai", "įžvalgos", "charakteristika", "отзыв",
        "рекомендация", "рецензия", "обзор", "мнение", "мысли", "оценка",
        "вклад", "комментарии", "соображения", "характеристика"
    ]
    return any(k in text_low for k in keywords)


def feature_subject_caps_ratio(msg):
    subject = msg.get("subject", "")
    letters = [c for c in subject if c.isalpha()]
    capitals = [c for c in subject if c.isupper()]
    return round(len(capitals) / len(letters), 4) if letters else 0


def feature_number_count(msg):
    return sum(c.isdigit() for c in msg["body"])


def feature_number_ratio(msg):
    letters = sum(c.isalpha() for c in msg["body"])
    digits = feature_number_count(msg)
    return round(digits / letters, 4) if letters else 0


def feature_finance_symbol_count(msg):
    return sum(msg["body"].count(sym) for sym in ["€", "$", "£", "¥"])


def feature_whitespace_ratio(msg):
    text = msg["body"]
    whites = sum(c.isspace() for c in text)
    letters = sum(c.isalpha() for c in text)
    return round(whites / letters, 4) if letters else 0


def feature_has_quoted_reply(msg):
    return ">" in msg["body"]


def feature_has_email_closing(msg):
    email_closings = [
        "Sincerely", "Yours sincerely", "Faithfully", "Yours faithfully", "Regards",
        "Best regards", "Kind regards", "Warm regards", "Best", "All the best",
        "Thank you", "Thanks", "Cheers", "Cordially", "Pagarbiai", "Su pagarba",
        "Nuoširdžiai", "Geriausi linkėjimai", "Linkėjimai", "Sėkmės", "Viso gero",
        "С уважением", "С наилучшими пожеланиями", "Всего доброго", "Всего хорошего",
        "Искренне Ваш", "Искренне Ваша", "С почтением", "Заранее спасибо",
    ]
    body = msg["body"].lower()
    for phrase in email_closings:
        if phrase.lower() in body:
            return True
    return False


# --- MAPPING OF THE FINAL 25 FEATURES ---
FEATURES = {
    # Core Content & Structure
    "from_domain": feature_from_domain,
    "body_word_count": feature_body_word_count,
    "subject_word_count": feature_subject_word_count,
    "longest_sentence_length": feature_longest_sentence_length,
    "has_quoted_reply": feature_has_quoted_reply,
    "has_email_closing": feature_has_email_closing,
    "link_count": feature_links,
    # Text Style & Formatting
    "capital_letter_count": feature_capitals,
    "capital_letter_ratio": feature_capital_ratio,
    "subject_caps_ratio": feature_subject_caps_ratio,
    "punctuation_count": feature_punctuation_count,
    "punctuation_ratio": feature_punctuation_ratio,
    "question_mark_count": feature_question_mark_count,
    "number_count": feature_number_count,
    "number_ratio": feature_number_ratio,
    "finance_symbol_count": feature_finance_symbol_count,
    "emoji_count": feature_emojis,
    "emoji_ratio": feature_emoji_ratio,
    # Keywords & Semantics
    "has_important": feature_has_important,
    "feedback_words": feature_feedback_words,
    # Metadata
    "day_of_week": feature_day_of_week,
    "time_of_day": feature_time_of_day,
    # Kept for total count
    "letter_count": feature_letters,
    "whitespace_ratio": feature_whitespace_ratio,
}

print(f"Number of curated features: {len(FEATURES)}")


def parse_email(path):
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    def get_header(name):
        return msg[name] or ""

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
        "from": from_, "to": to, "cc": cc, "subject": subject or "",
        "body": "\n".join(body_parts), "day_of_week": day_name, "time_of_day": time_seg,
    }


def build_record(msg, label):
    record = {}
    for name, func in FEATURES.items():
        try:
            record[name] = func(msg)
        except Exception:
            record[name] = None
    record["label"] = label
    return record


def main():
    labels = load_labels()
    # Assume EMAIL_DIR points to a folder with subfolders 'good' and 'spam' for labeling
    all_files = []
    for folder in ['good', 'spam']:
        folder_path = os.path.join(EMAIL_DIR, folder)
        if os.path.isdir(folder_path):
            all_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".eml")])

    records = []
    for path in all_files:
        msg_id = os.path.splitext(os.path.basename(path))[0]
        label = labels.get(msg_id)
        if not label:
            continue
        msg = parse_email(path)
        record = build_record(msg, label)
        records.append(record)

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
    print("This script is primarily intended to be used as a module by 'build_dataset.py'.")
    print("Running it standalone requires a 'emails_labeled.jsonl' file and a flat 'emails' directory.")
