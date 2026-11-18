#!/usr/bin/env python3
# FILE: build_name_txt.py
# PURPOSE: Multi-Dataset Zero-CVE Builder for Step 9:
#          - name_ask.txt (Class 23: ask_name)
#          - name_tell.txt (Class 24: tell_name)
#          - names_female_catalog.txt & names_male_catalog.txt (Multi-Dataset O(1) LUT Catalogs)
# RULES: Enforces Rules 1-13 (9 Parquet-only HF datasets, 0 .py scripts, 350+ English/Step collision blacklist).

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_LINE_RE = re.compile(r"^[a-z\s'?,]+$")
SPACE_RE = re.compile(r"\s+")

# Curated Global First Names (Female & Male: English, European, Arabic/Tunisian/MENA, Indian, Latin, Turkish, Asian)
CURATED_FEMALE_NAMES = (
    "emma", "sarah", "sara", "olivia", "ava", "sophia", "sofia", "isabella", "mia", "charlotte",
    "amelia", "harper", "evelyn", "abigail", "emily", "elizabeth", "mila", "ella", "avery", "camila",
    "aria", "scarlett", "anna", "leah", "natalie", "zoe", "zoey", "hannah", "hana", "addison",
    "lillian", "lucy", "paisley", "savannah", "claire", "skylar", "bella", "aubrey", "stella", "maya",
    "nora", "ellie", "samantha", "audrey", "allison", "alice", "madelyn", "hailey", "eva", "emilia",
    "quinn", "nevaeh", "piper", "everly", "cora", "kaylee", "lydia", "arianna", "eliana", "peyton",
    "melanie", "gianna", "isabelle", "julia", "valentina", "nova", "clara", "vivian", "reagan", "mackenzie",
    "madeline", "brielle", "delilah", "isla", "rylee", "katherine", "kate", "katie", "jessica", "jessie",
    "ashley", "amanda", "melissa", "nicole", "rachel", "lauren", "stephanie", "rebecca", "laura", "shannon",
    "angela", "kimberly", "amy", "michelle", "tiffany", "chloe", "jena", "jenna", "haley", "brooke",
    "paige", "molly", "vanessa", "erica", "elena", "nina", "maria", "marie", "carmen", "lucia",
    "paula", "marta", "daniela", "valeria", "camille", "lea", "manon", "juliette", "celine", "chiara",
    "giulia", "francesca", "alina", "daria",
    # Arabic / Tunisian / North African / Middle Eastern Female Names
    "nour", "meriem", "mariem", "maryam", "mariam", "salma", "selma", "ines", "yasmine", "yasmin",
    "yasmina", "fatima", "fatma", "amina", "amira", "leila", "layla", "laila", "lina", "hiba",
    "aya", "eyaa", "eya", "farah", "rim", "rima", "reem", "rana", "nada", "nadia",
    "nesrine", "nisrine", "syrine", "sirine", "cyrine", "chaima", "shayma", "soumaya", "somaya", "asma",
    "ikram", "imen", "imane", "iman", "amal", "amel", "ons", "malak", "malek", "marwa",
    "maroua", "safa", "sana", "samar", "sahar", "zeineb", "zaynab", "zainab", "khadija", "halima",
    "soukaina", "kenza", "ghita", "dounia", "donia", "habiba", "rahma", "rahmah", "sihem", "houda",
    "huda", "hend", "hinda", "nourhene", "jihene", "feriel", "bochra", "bouchra", "tasnim", "kawthar",
    # Indian / Turkish / Asian Female Names
    "priya", "ananya", "divya", "pooja", "neha", "riya", "shreya", "sneha", "kavya", "aisha",
    "aishwarya", "anjali", "kriti", "tanvi", "nisha", "meera", "simran", "zoya", "diya", "ishi",
    "elif", "zeynep", "defne", "azra", "ecrin", "nehir", "eylul", "yagmur", "merve", "busra",
    "yuki", "sakura", "mei", "minji", "jisoo", "yuna", "sora", "mina",
)

CURATED_MALE_NAMES = (
    "alex", "liam", "noah", "oliver", "elijah", "william", "james", "benjamin", "lucas", "henry",
    "alexander", "michael", "ethan", "daniel", "jacob", "logan", "jackson", "levi", "sebastian", "mateo",
    "jack", "owen", "theodore", "aiden", "samuel", "joseph", "john", "david", "wyatt", "matthew",
    "luke", "asher", "julian", "grayson", "leo", "jayden", "gabriel", "isaac", "lincoln", "anthony",
    "hudson", "dylan", "ezra", "thomas", "charles", "christopher", "jaxon", "maverick", "josiah", "isaiah",
    "andrew", "elias", "joshua", "nathan", "caleb", "ryan", "adrian", "eli", "nolan", "aaron",
    "cameron", "ezekiel", "colton", "luca", "landon", "jonathan", "santiago", "axel", "easton", "jeremiah",
    "connor", "jameson", "robert", "greyson", "ian", "carson", "jaxson", "leonardo", "nicholas", "dominic",
    "everett", "xavier", "kai", "jose", "adam", "jace", "wesley", "kayden", "silas", "bennett",
    "declan", "waylon", "weston", "evan", "emmett", "micah", "ryder", "beau", "damian", "brayden",
    "gael", "rowan", "harrison", "bryson", "amir", "kingston", "jason", "justin", "brandon", "kevin",
    "brian", "eric", "steven", "timothy", "jeffrey", "scott", "jake", "chris", "matt", "nick",
    "zach", "zack", "cody", "kyle", "tyler", "max", "ben", "tom", "dan", "dave",
    "joe", "carlos", "diego", "luis", "jorge",
    # Arabic / Tunisian / North African / Middle Eastern Male Names
    "hatem", "omar", "ahmed", "ahmad", "mohamed", "mohammed", "muhammad", "ali", "youssef", "yousef",
    "yusuf", "amine", "amin", "yassine", "yassin", "yasine", "mehdi", "mahdi", "karim", "kareem",
    "rami", "ramy", "walid", "waleed", "bilel", "bilal", "hamza", "aymen", "ayman", "anas",
    "zied", "ziad", "ziyad", "rayen", "rayan", "tarak", "tarek", "tariq", "farouk", "iskander",
    "skander", "achraf", "ashraf", "sofiane", "soufiane", "nabil", "mourad", "murad", "adel", "adnan",
    "samir", "sami", "samy", "taha", "marwan", "marouane", "ousama", "oussama", "osama", "khalil",
    "khalid", "khaled", "slim", "salim", "selim", "kais", "qais", "firas", "fadi", "hassan",
    "hassen", "hussein", "houssem", "hichem", "hicham", "rachid", "rashid", "said", "saeed", "majdi",
    "mejdi", "makram", "wissem", "wassim", "wael", "wail", "ibrahim", "ismail", "zakaria",
    # Indian / Turkish / Asian Male Names
    "rahul", "rohit", "arjun", "aarav", "vihaan", "aditya", "aryan", "sai", "krishna", "ishaan",
    "shaurya", "atharv", "dhruv", "kabir", "ritvik", "ayaan", "kunal", "varun", "nikhil", "aman",
    "eymen", "omer", "mirac", "berat", "ahmet", "mehmet", "mustafa", "emre", "burak",
    "kenji", "haruto", "ren", "hiro", "minho", "jihoon", "taehyung", "jun", "wei", "chen",
)

# Ultra-Strict Blacklist: Blocks all Steps 1-8 tokens, Countries, Cities, and 350+ Common English words that double as names
NAME_COLLISION_BLACKLIST = frozenset((
    # Step 1-8 Core Vocabulary
    "sam", "same", "hi", "hey", "hello", "helo", "hlo", "yo", "sup", "howdy", "welcome", "aloha",
    "morning", "mornin", "afternoon", "evening", "night", "day", "gday", "greetings", "salutations",
    "age", "old", "olds", "from", "country", "where", "which", "live", "living", "year", "years",
    "how", "hows", "hru", "hyd", "hor", "wbu", "hbu", "wyd", "wud", "whatcha",
    "doing", "doin", "diong", "dopin", "going", "goin", "good", "great", "fine", "ok", "okay",
    "alright", "well", "happy", "awesome", "amazing", "fantastic", "bored", "borad", "boerd", "borde",
    "tired", "tierd", "sleepy", "exhausted", "sad", "lonely", "bad", "sick", "stressed",
    "nothing", "nothin", "nada", "nm", "nmu", "chat", "chatting", "chatin", "talking",
    "chilling", "chillin", "relaxing", "resting", "bed", "laying", "lying", "home", "working",
    "work", "busy", "studying", "reading", "listening", "music", "watching", "movie", "movies",
    "youtube", "netflix", "gaming", "playing", "games", "game", "scrolling", "phone", "eating",
    "cooking", "drinking", "coffee", "cute", "cuute", "cutie", "sweet", "sweat", "nice", "niice",
    "cool", "coool", "hot", "hott", "pretty", "prety", "gorgeous", "beautiful", "handsome", "sexy",
    "attractive", "adorable", "lovely", "charming", "funny", "fun", "smart", "kind", "chill",
    "like", "luv", "love", "vibe", "energy", "smile", "style", "type", "blush", "speaking",
    "haha", "hahaha", "hehe", "lol", "lmao", "lmfao", "rofl", "dope", "lit", "fire", "bet",
    "word", "fair", "true", "tru", "facts", "fax", "fr", "ikr", "right", "real", "agreed",
    "agree", "totally", "definitely", "exact", "exactly", "kk", "okey", "oki", "okie", "alr",
    "aight", "yeah", "yea", "yep", "yup", "yas", "yass", "sure", "ofc", "aha", "wow", "omg",
    "looking", "lookin", "loking", "lokin", "finding", "findin", "find", "seeking", "seekin", "seek",
    "searching", "searchin", "search", "brings", "bring", "goal", "purpose", "reason",
    "gf", "gfs", "girlfriend", "grilfriend", "wifey", "wife", "bf", "bfs", "boyfriend", "hubby", "husband",
    "relationship", "relationships", "partner", "soulmate", "dating", "date", "dates", "romance",
    "romantic", "flirting", "flirt", "hookup", "friends", "friend", "freind", "freinds", "frends",
    "friendship", "bestie", "company", "girl", "gril", "gurl", "girls", "grils", "woman", "women",
    "female", "famle", "famel", "females", "lady", "ladies", "boy", "boi", "boys", "guy", "guys",
    "man", "men", "male", "maale", "males", "dude", "dudes", "gender", "gander", "sex",
    "making", "make", "trying", "tryna", "want", "wan", "wanna", "wanting", "need", "needing",
    "hoping", "meet", "meeting", "someone", "somebody", "people", "ppl", "connection",
    # Countries & Cities from Step 2
    "tunisia", "tunis", "tuinsi", "tunsia", "tounsi", "sfax", "sousse", "india", "indian", "hindi",
    "handi", "desi", "bharat", "mumbai", "delhi", "usa", "america", "american", "states", "murica",
    "cali", "california", "texas", "florida", "uk", "britain", "british", "england", "english",
    "brit", "london", "scotland", "algeria", "dziri", "algiers", "oran", "morocco", "maroc",
    "moroccan", "maghreb", "casa", "casablanca", "rabat", "egypt", "egyptian", "masr", "misr",
    "cairo", "alexandria", "france", "french", "paris", "lyon", "germany", "german", "berlin",
    "italy", "italia", "italian", "rome", "milan", "spain", "espana", "spanish", "madrid",
    "canada", "canadian", "toronto", "saudi", "riyadh", "arabia", "uae", "dubai", "emirates",
    "emirati", "abudhabi", "turkey", "turk", "turkiye", "turkish", "istanbul", "pakistan", "paki",
    "pakistani", "lahore", "karachi", "indonesia", "indo", "indonesian", "bali", "jakarta",
    "philippines", "pinoy", "pinay", "filipino", "filipina", "manila", "brazil", "brasil",
    "brazilian", "rio", "jordan", "chad", "georgia", "sydney", "brooklyn", "austin", "dallas",
    "victoria", "sofia", "lima", "china", "chinese", "japan", "japanese", "tokyo", "korea", "korean",
    # Common English words that appear in baby-name registries
    "will", "may", "can", "art", "bill", "bob", "mark", "rob", "rich", "frank", "gene", "pat",
    "grace", "hope", "faith", "joy", "summer", "autumn", "winter", "spring", "april", "june",
    "july", "august", "monday", "friday", "sunday", "sunny", "star", "sky", "rain", "snow",
    "river", "ocean", "forest", "meadow", "dawn", "eve", "angel", "heaven", "destiny", "harmony",
    "melody", "trinity", "serenity", "charity", "chastity", "mercy", "justice", "liberty",
    "freedom", "peace", "truth", "honor", "glory", "royal", "king", "queen", "prince", "princess",
    "duke", "earl", "lord", "sir", "master", "saint", "major", "general", "captain", "marshal",
    "dean", "mason", "carter", "hunter", "fisher", "baker", "miller", "taylor", "walker", "cook",
    "cooper", "parker", "turner", "ward", "wood", "woods", "stone", "rock", "hill", "dale",
    "glen", "brook", "brooks", "lane", "page", "reed", "reid", "cole", "clay", "grant", "chase",
    "chance", "cash", "gage", "miles", "west", "north", "east", "south", "young", "new", "poor",
    "wise", "wild", "free", "noble", "brave", "bold", "strong", "keen", "sharp", "swift", "fast",
    "slow", "still", "calm", "quiet", "soft", "warm", "cold", "dry", "wet", "red", "blue",
    "green", "white", "black", "brown", "gray", "grey", "gold", "golden", "silver", "rose",
    "violet", "lily", "daisy", "iris", "ivy", "holly", "hazel", "olive", "fawn", "robin", "wren",
    "lark", "raven", "jay", "drake", "wolf", "bear", "fox", "colt", "buck", "hart", "ray", "beam",
    "what", "whats", "wat", "wht", "why", "who", "when", "this", "that", "here", "there",
    "name", "names", "named", "nam", "nme", "naem", "nickname", "call", "called", "cal",
    "tell", "know", "ask", "get", "should", "first", "please", "plz", "help", "time", "want",
    "genesis", "ruby", "amber", "jade", "pearl", "diamond", "crystal", "opal", "coral",
    "christian", "roman", "sawyer", "lin", "jun",
))

CLEAN_F_NAMES = tuple(sorted({w.strip().lower() for w in CURATED_FEMALE_NAMES if w.strip().lower().isalpha() and w.strip().lower() not in NAME_COLLISION_BLACKLIST}))
CLEAN_M_NAMES = tuple(sorted({w.strip().lower() for w in CURATED_MALE_NAMES if w.strip().lower().isalpha() and w.strip().lower() not in NAME_COLLISION_BLACKLIST and w.strip().lower() not in CLEAN_F_NAMES}))

STRICT_ASK_NAME_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+|w+b+u+|h+b+u+|a+n+d+)\s+)?"
    r"(?:"
    r"(?:w+h+a+t+'?s|w+h+a+t+s|w+h+a+t+\s+i+s+|w+a+t+\s+i+s+|w+a+t+s)\s+(?:y+o+u+r+|u+r+|y+a+)\s+"
    r"(?:f+i+r+s+t+\s+|r+e+a+l+\s+)?(?:n+a+m+e+|n+a+m+|n+m+e+|n+a+e+m+|n+i+c+k+n+a+m+e+)|"
    r"(?:y+o+u+r+|u+r+)\s+(?:n+a+m+e+|n+a+m+|n+m+e+)\s*\?*|"
    r"(?:n+a+m+e+|n+a+m+)\s*\?+|"
    r"(?:w+h+a+t+|w+a+t+)\s+(?:s+h+o+u+l+d+|c+a+n+|d+o+)\s+i+\s+c+a+l+l+\s+(?:y+o+u+|u+)|"
    r"(?:w+h+a+t+|w+a+t+)\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+)\s+c+a+l+l+e+d+|"
    r"(?:m+a+y+|c+a+n+)\s+i+\s+(?:k+n+o+w+|a+s+k+|h+a+v+e+|g+e+t+)\s+(?:y+o+u+r+|u+r+)\s+(?:n+a+m+e+|n+a+m+)|"
    r"t+e+l+l+\s+m+e+\s+(?:y+o+u+r+|u+r+)\s+(?:n+a+m+e+|n+a+m+)"
    r")(?:\s+(?:c+u+t+i+e+|p+l+z+|p+l+e+a+s+e+))?\??$",
    re.IGNORECASE,
)

ASK_NAME_BASES = (
    "what is your name", "whats your name", "what's your name", "whats ur name", "what's ur name",
    "what is ur name", "wat is ur name", "wats ur name", "whats ya name",
    "whats your first name", "whats ur first name", "whats your real name", "whats ur real name",
    "whats ur nam", "whats ur nme", "whats ur naem", "whats your nickname",
    "your name?", "ur name?", "your name", "ur name", "ur nam?", "name?", "nam?",
    "what should i call you", "what should i call u", "what can i call you", "what can i call u",
    "what do i call u", "what are you called", "what r u called",
    "may i know your name", "may i know ur name", "can i know your name", "can i know ur name",
    "can i ask your name", "can i ask ur name", "can i get ur name",
    "tell me your name", "tell me ur name", "whats ur name cutie",
)

NAME_TELL_FRAMES = (
    "my name is {name}",
    "my name's {name}",
    "name is {name}",
    "im {name}",
    "i'm {name}",
    "lm {name}",
    "i am {name}",
    "call me {name}",
    "u can call me {name}",
    "you can call me {name}",
    "its {name}",
    "it's {name}",
    "hi im {name}",
    "hey im {name}",
)

EXTRACT_HF_NAME_RE = re.compile(
    r"\b(?:my\s+name\s+is|my\s+name's|call\s+me|i\s+am|i'm)\s+([A-Z][a-z]{3,12})\b"
)


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    if valid.shape[0] == 0:
        return ""
    return valid.tobytes().decode("ascii", errors="ignore")


def _is_valid_external_name(raw_name):
    if not isinstance(raw_name, str):
        return ""
    w = _sanitize_ascii_bytes(raw_name).strip().lower()
    if not (4 <= len(w) <= 13) or not w.isalpha():
        return ""
    if w in NAME_COLLISION_BLACKLIST:
        return ""
    return w


def _collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def _canonical_token_sig(s):
    has_q = "?" in s
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, has_q)


def _stream_all_parquet_name_datasets(ask_map, mined_f, mined_m):
    """
    Rule 12 Multi-Dataset Parquet Harvester:
    Streams from 9 HuggingFace datasets using pure Parquet (zero remote .py scripts!)
    """
    stats = {"datasets_hit": 0, "names_mined": 0, "dialogs_scanned": 0}
    try:
        from datasets import load_dataset
    except ImportError:
        return stats

    # A) Dedicated First-Name & Gender Datasets (Parquet)
    name_repos = (
        ("takiuddinahmed/muslim-names-dataset", None, "refs/convert/parquet", ("name", "Name"), ("gender", "Gender"), 2500),
        ("erickrribeiro/gender-by-name", None, "refs/convert/parquet", ("name", "Name"), ("gender", "sex"), 2500),
        ("aborasheed/Global-Human-Names", None, None, ("first_name", "name", "First_Name"), ("gender", "sex"), 3000),
        ("nkrishnaswami/us-ssa-baby-names-national", None, None, ("name",), ("sex",), 2500),
        ("onomaverse/names", "name-gender-inference", None, ("name", "first_name"), ("gender", "sex"), 2500),
    )

    for repo, cfg, rev, name_cols, gen_cols, max_rows in name_repos:
        try:
            kwargs = {"split": "train", "streaming": True}
            if rev:
                kwargs["revision"] = rev
            ds = load_dataset(repo, cfg, **kwargs) if cfg else load_dataset(repo, **kwargs)
            stats["datasets_hit"] += 1
            c = 0
            for row in ds:
                if c >= max_rows:
                    break
                c += 1
                raw_n = ""
                for nc in name_cols:
                    if nc in row and row[nc]:
                        raw_n = str(row[nc])
                        break
                clean_n = _is_valid_external_name(raw_n)
                if not clean_n:
                    continue
                raw_g = ""
                for gc in gen_cols:
                    if gc in row and row[gc] is not None:
                        raw_g = str(row[gc]).strip().lower()
                        break
                if raw_g in ("f", "female", "girl", "woman", "1"):
                    if clean_n not in mined_m:
                        mined_f.add(clean_n)
                        stats["names_mined"] += 1
                else:
                    if clean_n not in mined_f:
                        mined_m.add(clean_n)
                        stats["names_mined"] += 1
        except Exception:
            pass

    # B) Multi-Dataset Dialogue Streams (daily_dialog, personachat, empathetic_dialogues)
    dialog_repos = (
        ("daily_dialog", "refs/convert/parquet", "dialog", 2500),
        ("bavard/personachat_truecased", "refs/convert/parquet", "history", 1500),
        ("AlekseyKorshuk/persona-chat", "refs/convert/parquet", "utterances", 1500),
        ("empathetic_dialogues", "refs/convert/parquet", "utterance", 1500),
    )

    for repo, rev, col, max_rows in dialog_repos:
        try:
            ds = load_dataset(repo, split="train", revision=rev, streaming=True)
            stats["datasets_hit"] += 1
            c = 0
            for row in ds:
                if c >= max_rows:
                    break
                c += 1
                stats["dialogs_scanned"] += 1
                val = row.get(col, ())
                if isinstance(val, str):
                    turns = (val,)
                elif isinstance(val, (list, tuple)):
                    turns = val
                else:
                    continue
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    clean_turn = _sanitize_ascii_bytes(turn)
                    for m in EXTRACT_HF_NAME_RE.finditer(clean_turn):
                        cand_n = _is_valid_external_name(m.group(1))
                        if cand_n and cand_n not in mined_f and cand_n not in mined_m:
                            mined_m.add(cand_n)
                            stats["names_mined"] += 1
                    for sent in re.split(r"[.!]+", clean_turn[:90]):
                        s = SPACE_RE.sub(" ", sent.strip().lower()).rstrip("!.,")
                        if s and len(s) <= 40 and CLEAN_LINE_RE.match(s) and STRICT_ASK_NAME_RE.match(s):
                            sig = _canonical_token_sig(s)
                            if sig not in ask_map:
                                ask_map[sig] = s
        except Exception:
            pass

    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_name_files(
    ask_path="name_ask.txt",
    tell_path="name_tell.txt",
    f_catalog_path="names_female_catalog.txt",
    m_catalog_path="names_male_catalog.txt",
    neg_path="negatives.txt",
):
    ask_map = {}
    tell_map = {}
    mined_f = set(CLEAN_F_NAMES)
    mined_m = set(CLEAN_M_NAMES)

    for base in ASK_NAME_BASES:
        if "?" in base:
            s = base.strip().lower()
            ask_map[_canonical_token_sig(s)] = s
            continue
        for pfx in ("", "hey", "so", "btw", "wbu"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cand = f"{full}{end}"
                if STRICT_ASK_NAME_RE.match(cand):
                    sig = _canonical_token_sig(cand)
                    if sig not in ask_map:
                        ask_map[sig] = cand

    hf_stats = _stream_all_parquet_name_datasets(ask_map, mined_f, mined_m)

    mined_f.difference_update(NAME_COLLISION_BLACKLIST)
    mined_m.difference_update(NAME_COLLISION_BLACKLIST | mined_f)

    # Save complete Female & Male O(1) LUT catalogs for step9_name.py
    n_f_cat, b_f_cat = _write_u8_file(f_catalog_path, mined_f)
    n_m_cat, b_m_cat = _write_u8_file(m_catalog_path, mined_m)

    all_catalog_names = sorted(mined_f | mined_m)
    core_sample_names = (
        "emma", "sarah", "sara", "olivia", "sophia", "mia", "chloe", "charlotte", "emily", "jessica",
        "nour", "meriem", "salma", "ines", "yasmine", "fatima", "amina", "leila", "lina", "priya",
        "alex", "liam", "noah", "jake", "jack", "james", "john", "david", "michael", "daniel",
        "hatem", "omar", "ahmed", "mohamed", "ali", "youssef", "amine", "yassine", "mehdi", "karim",
        "rami", "walid", "bilel", "hamza", "aymen", "anas", "zied", "rayen", "tarak", "rahul",
    )

    # 1) All catalog names (up to 2,500) as bare & "im <name>", "my name is <name>" for MLP training
    for nm in all_catalog_names[:2500]:
        for cand in (nm, f"im {nm}", f"my name is {nm}"):
            sig = _canonical_token_sig(cand)
            if sig not in tell_map:
                tell_map[sig] = cand

    # 2) Richer conversational framing + ask-backs on core names + "sam" (framed only!)
    for nm in core_sample_names + ("sam",):
        for tmpl in NAME_TELL_FRAMES:
            base = tmpl.format(name=nm)
            for sfx in ("", " u?", " , u", " wbu?"):
                cand = f"{base}{sfx}".strip()
                if len(cand) <= 42 and len(cand.split()) <= 6:
                    sig = _canonical_token_sig(cand)
                    if sig not in tell_map:
                        tell_map[sig] = cand

    for k_sig in list(tell_map.keys()):
        if k_sig in ask_map:
            del tell_map[k_sig]

    n_ask, b_ask = _write_u8_file(ask_path, ask_map.values())
    n_tell, b_tell = _write_u8_file(tell_path, tell_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in ask_map.values():
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    for nm in core_sample_names:
        existing_neg.add(nm)
        existing_neg.add(f"my name is {nm}"[:30])
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[MULTI-DATASET PARQUET HARVESTER] Hit {hf_stats['datasets_hit']} HF Parquet datasets | "
        f"Mined {hf_stats['names_mined']} extra names & scanned {hf_stats['dialogs_scanned']} dialogues!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 9 CATALOGS] Indexed {n_f_cat} Female names ({b_f_cat / 1024.0:.2f} KB) "
        f"& {n_m_cat} Male names ({b_m_cat / 1024.0:.2f} KB)",
        flush=True,
    )
    print(
        f"[CLEAN STEP 9 TRAINING] Saved {n_ask} ask lines to '{ask_path}' ({b_ask / 1024.0:.2f} KB) "
        f"& {n_tell} tell lines to '{tell_path}' ({b_tell / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_name_files()