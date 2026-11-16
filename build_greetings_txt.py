# FILE: build_greetings_txt.py
# PURPOSE: Build a clean, deduplicated English greetings.txt (25k+ real chat variations)

import re

CLEAN_RE = re.compile(r"^[a-zA-Z\s'!?.,:;)-]+$")

ROOT_GREETINGS = (
    "hi", "hello", "hey", "hiya", "howdy", "yo", "sup", "ello", "heya", "hullo",
    "greetings", "salutations", "welcome", "aloha", "gday", "g day",
    "good morning", "morning", "mornin", "good afternoon", "afternoon",
    "good evening", "evening", "good day", "top of the morning",
    "whats up", "what is up", "wassup", "wazzup", "whazzup", "sup bro", "sup man",
    "how are you", "how r u", "how are ya", "how u doing", "how are you doing",
    "hows it going", "how is it going", "hows everything", "how have you been",
    "how do you do", "how is your day", "hows ur day", "hru", "hyd",
    "nice to meet you", "pleased to meet you", "glad to meet you", "good to see you",
    "long time no see", "hey there", "hi there", "hello there", "yo there",
    "hey you", "hi you", "hello friend", "hey friend", "hey buddy", "hey mate",
    "hey stranger", "hi stranger", "hello stranger", "hey everyone", "hi all",
)

ELONGATED_WORDS = (
    "hi", "hii", "hiii", "hiiii", "hiiiii",
    "hey", "heyy", "heyyy", "heyyyy", "heyyyyy",
    "hello", "helloo", "hellooo", "helloooo",
    "yo", "yoo", "yooo", "yoooo",
    "sup", "supp", "suppp",
    "hiya", "hiyaa", "hiyaaa",
    "heya", "heyaa", "heyaaa",
    "wassup", "wassupp", "wassuppp",
    "morning", "morningg", "morninn",
)

SHORT_OPENERS = (
    "", "there", "u", "you", "friend", "stranger", "cutie", "gorgeous", "handsome",
    "hru", "hyd", "whats up", "wassup", "how r u", "how are you", "hows it going",
    "nice to meet u", "nice to meet you", "how is ur day", "good morning", "good evening",
)

ENDINGS = ("", "!", "!!", "!!!", ".", "..", "?", " :)", " :D", " ;)", " <3", " ^^")


def is_valid_greeting(text):
    s = text.strip().lower()
    if not s or len(s) > 30 or len(s.split()) > 4:
        return ""
    if not CLEAN_RE.match(s):
        return ""
    return s


def pull_huggingface_greetings(out_set):
    try:
        from datasets import load_dataset
    except ImportError:
        return

    sources = (
        ("AmazonScience/massive", "en-US", "train", "utt", "intent", "general_greet"),
        ("clinc/clinc_oos", "plus", "train", "text", "intent", "greeting"),
    )
    for repo, cfg, split, txt_col, lbl_col, target_name in sources:
        try:
            ds = load_dataset(repo, cfg, split=split)
            feat = ds.features[lbl_col]
            for row in ds:
                if feat.int2str(row[lbl_col]) == target_name:
                    cleaned = is_valid_greeting(row[txt_col])
                    if cleaned:
                        out_set.add(cleaned)
        except Exception:
            pass

    try:
        ds_tanaos = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1", split="train")
        for row in ds_tanaos:
            if int(row["label"]) == 0:
                cleaned = is_valid_greeting(row["text"])
                if cleaned:
                    out_set.add(cleaned)
    except Exception:
        pass


def generate_greetings_file(output_path="greetings.txt"):
    unique_lines = set()
    pull_huggingface_greetings(unique_lines)

    for base in ROOT_GREETINGS:
        for end in ENDINGS:
            cleaned = is_valid_greeting(f"{base}{end}")
            if cleaned:
                unique_lines.add(cleaned)

    for word in ELONGATED_WORDS:
        for opener in SHORT_OPENERS:
            phrase = f"{word} {opener}".strip()
            for end in ENDINGS:
                cleaned = is_valid_greeting(f"{phrase}{end}")
                if cleaned:
                    unique_lines.add(cleaned)

    sorted_lines = sorted(unique_lines)
    with open(output_path, "w", encoding="ascii") as f:
        for line in sorted_lines:
            f.write(line + "\n")

    print(f"[DONE] Wrote {len(sorted_lines)} clean, unique English greetings to '{output_path}'")


if __name__ == "__main__":
    generate_greetings_file()