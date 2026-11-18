#!/usr/bin/env python3
# FILE: build_hobbies_txt.py
# PURPOSE: Zero-CVE Multi-Dataset Builder for Step 10:
#          - hobbies_ask.txt (Class 25: ask_hobbies_interests)
#          - hobbies_tell.txt (Class 26: tell_hobbies_interests)
# RULES: Enforces Rules 1-13 (Parquet-only fetch, 7-bit ASCII byte strip, closed vocab whitelist,
#        protects Step 6 "listening to music"/"watching a movie"/"gaming" & Class 0 "what do you do").

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

CLEAN_LINE_RE = re.compile(r"^[a-z0-9\s'?,]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard:
# Protects Step 6 present-continuous status phrases ("listening to music", "watching tv", "gaming")
# Protects Step 7 ("i like u", "love ur vibe"), Step 8 ("looking for fun"), and Class 0 ("what do you do")
STEP1_TO_9_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:w+h+a+t+\s+d+o+\s+(?:y+o+u+|u+)\s+d+o+)|"
    r"(?:(?:i+m+|i+'?m+|l+m+|j+u+s+t+)?\s*(?:"
    r"l+i+s+t+e+n+i+n+g+\s+(?:t+o+\s+)?m+u+s+i+c+|"
    r"w+a+t+c+h+i+n+g+\s+(?:t+v+|a+\s+m+o+v+i+e+|m+o+v+i+e+s+|y+o+u+t+u+b+e+|n+e+t+f+l+i+x+)|"
    r"g+a+m+i+n+g+|p+l+a+y+i+n+g+\s+(?:g+a+m+e+s+|a+\s+g+a+m+e+)|"
    r"r+e+a+d+i+n+g+|c+o+o+k+i+n+g+|s+t+u+d+y+i+n+g+|w+o+r+k+i+n+g+"
    r")(?:\s+(?:r+n+|n+o+w+|u+|w+b+u+))?)|"
    r"(?:(?:i+\s+)?(?:r+e+a+l+y+\s+)?(?:l+i+k+e+|l+o+v+e+)\s+(?:y+o+u+|u+|t+a+l+k+i+n+g+\s+t+o+\s+(?:y+o+u+|u+)|(?:y+o+u+r+|u+r+)\s+v+i+b+e+))|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|g+f+|b+f+|g+i+r+l+f+r+i+e+n+d+|b+o+y+f+r+i+e+n+d+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b.*|"
    r".*\b(?:h+r+u+|h+y+d+|w+y+d+|w+u+d+|b+o+r+e+d+|b+o+r+a+d+|t+i+r+e+d+|c+h+i+l+i+n+g*|n+m+u+)\b.*"
    r")[\s?]*$",
    re.IGNORECASE,
)

# 5 Hobby & Interest Category Vocabularies:
# 1 = Music, 2 = Gaming, 3 = Movies/TV/Anime, 4 = Sports/Gym, 5 = General Hobbies/Travel/Art/Books
S10_MUSIC_WORDS = (
    "music", "song", "songs", "rap", "hiphop", "pop", "rock", "rnb", "kpop",
    "edm", "jazz", "metal", "indie", "country", "techno", "house", "lofi",
    "singer", "artist", "band", "spotify", "singing", "sing", "guitar", "piano",
)
S10_GAME_WORDS = (
    "game", "games", "gaming", "gamer", "valorant", "fortnite", "minecraft",
    "roblox", "cod", "warzone", "fifa", "gta", "league", "pubg", "apex",
    "overwatch", "csgo", "genshin", "elden", "ps5", "ps4", "xbox", "pc", "nintendo",
)
S10_MOVIE_WORDS = (
    "movie", "movies", "film", "films", "anime", "manga", "show", "shows",
    "series", "kdrama", "netflix", "youtube", "tv", "horror", "comedy",
    "action", "thriller", "scifi", "romance", "marvel", "disney",
)
S10_SPORT_WORDS = (
    "sport", "sports", "gym", "workout", "workouts", "fitness", "lifting",
    "football", "soccer", "basketball", "tennis", "volleyball", "baseball",
    "swimming", "running", "jogging", "boxing", "mma", "ufc", "cycling", "yoga",
)
S10_GENERAL_HOBBY_WORDS = (
    "hobby", "hobbies", "hoby", "hobies", "interest", "interests", "passion",
    "travel", "traveling", "travelling", "hiking", "camping", "beach", "nature",
    "drawing", "art", "painting", "sketching", "photography", "photos",
    "reading", "books", "book", "novels", "writing", "cooking", "baking",
    "dancing", "dance", "shopping", "fashion", "cars", "bikes", "motorcycles",
    "coding", "programming", "tech", "dogs", "cats", "pets", "animals",
)

S10_VERB_PREF_WORDS = (
    "like", "likes", "love", "loves", "enjoy", "enjoys", "into", "fan",
    "listen", "listening", "play", "playing", "watch", "watching",
    "favorite", "favourite", "fav", "prefer",
)
S10_QUESTION_WORDS = (
    "what", "whats", "what's", "wat", "wht", "which", "any", "kind", "type", "genre",
)
S10_FRAME_WORDS = (
    "do", "does", "are", "ar", "r", "is", "you", "u", "ya", "your", "ur",
    "i", "im", "i'm", "lm", "am", "my", "me", "to", "of", "for", "in", "on",
    "free", "spare", "time", "fun", "doing", "go", "going", "a", "an", "the",
    "lot", "mostly", "really", "so", "much", "some", "good", "new", "video",
    "working", "out", "and", "or", "wbu", "hbu", "btw", "hi", "hey",
)

STRICT_ALLOWED_S10_VOCAB = frozenset(
    S10_MUSIC_WORDS + S10_GAME_WORDS + S10_MOVIE_WORDS + S10_SPORT_WORDS
    + S10_GENERAL_HOBBY_WORDS + S10_VERB_PREF_WORDS + S10_QUESTION_WORDS + S10_FRAME_WORDS
)

# Strictly matches Class 25 (ask_hobbies_interests)
STRICT_ASK_HOBBIES_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+|w+b+u+|h+b+u+|a+n+d+)\s+)?"
    r"(?:"
    r"(?:w+h+a+t+|w+a+t+|w+h+t+|w+h+i+c+h+)(?:\s+(?:k+i+n+d+|t+y+p+e+|g+e+n+r+e+)\s+o+f+)?\s+"
    r"(?:m+u+s+i+c+|s+o+n+g+s*|g+a+m+e+s*|v+i+d+e+o+\s+g+a+m+e+s*|m+o+v+i+e+s*|f+i+l+m+s*|"
    r"a+n+i+m+e+|s+h+o+w+s*|s+e+r+i+e+s+|s+p+o+r+t+s*|h+o+b+i+e+s*|b+o+o+k+s*)\s+"
    r"(?:d+o+\s+|a+r+e+\s+|r+\s+)?(?:y+o+u+|u+)\s+"
    r"(?:l+i+k+e+|l+o+v+e+|l+i+s+t+e+n+(?:\s+t+o+)?|p+l+a+y+|w+a+t+c+h+|r+e+a+d+|i+n+t+o+)|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s|w+h+a+t+\s+i+s+|w+h+a+t+\s+a+r+e+|w+h+a+t+\s+r+)\s+"
    r"(?:y+o+u+r+|u+r+)\s+(?:f+a+v+o*u*r+i+t+e+\s+|f+a+v+\s+)?"
    r"(?:h+o+b+i+e+s*|h+o+b+y+|i+n+t+e+r+e+s+t+s*|m+u+s+i+c+|s+o+n+g+|g+a+m+e+s*|m+o+v+i+e+s*|a+n+i+m+e+|s+h+o+w+|s+p+o+r+t+)|"
    r"(?:w+h+a+t+|w+a+t+)\s+d+o+\s+(?:y+o+u+|u+)\s+(?:d+o+\s+f+o+r+\s+f+u+n+|d+o+\s+i+n+\s+(?:y+o+u+r+|u+r+)\s+f+r+e+e+\s+t+i+m+e+|"
    r"l+i+k+e+\s+(?:t+o+\s+d+o+|d+o+i+n+g+)(?:\s+f+o+r+\s+f+u+n+)?)|"
    r"(?:d+o+|a+r+e+|r+)\s+(?:y+o+u+|u+)\s+(?:l+i+k+e+|l+o+v+e+|p+l+a+y+|w+a+t+c+h+|l+i+s+t+e+n+\s+t+o+|i+n+t+o+)\s+"
    r"(?:m+u+s+i+c+|r+a+p+|p+o+p+|r+o+c+k+|k+p+o+p+|g+a+m+e+s*|v+i+d+e+o+\s+g+a+m+e+s*|g+a+m+i+n+g+|"
    r"v+a+l+o+r+a+n+t+|f+o+r+t+n+i+t+e+|m+i+n+e+c+r+a+f+t+|m+o+v+i+e+s*|a+n+i+m+e+|k+d+r+a+m+a+|"
    r"s+p+o+r+t+s*|f+o+o+t+b+a+l+l+|s+o+c+c+e+r+|b+a+s+k+e+t+b+a+l+l+|g+y+m+|w+o+r+k+i+n+g+\s+o+u+t+|"
    r"t+r+a+v+e+l+l*i+n+g*|r+e+a+d+i+n+g+|c+o+o+k+i+n+g+|d+r+a+w+i+n+g+)|"
    r"(?:a+n+y+\s+h+o+b+i+e+s*|y+o+u+r+\s+h+o+b+i+e+s*|u+r+\s+h+o+b+i+e+s*|h+o+b+i+e+s+\?+|"
    r"m+u+s+i+c+\?+|g+a+m+e+s+\?+|a+n+i+m+e+\?+|m+o+v+i+e+s+\?+|s+p+o+r+t+s+\?+)"
    r")\??$",
    re.IGNORECASE,
)

# Strictly matches Class 26 (tell_hobbies_interests)
STRICT_TELL_HOBBIES_RE = re.compile(
    r"^(?:"
    r"(?:(?:i+|i+m+|i+'?m+|l+m+|i+\s+a+m+)\s+)?(?:r+e+a+l+y+\s+|m+o+s+t+l+y+\s+)?"
    r"(?:l+i+k+e+|l+o+v+e+|e+n+j+o+y+|i+n+t+o+|p+r+e+f+e+r+|l+i+s+t+e+n+\s+t+o+|p+l+a+y+|w+a+t+c+h+)\s+"
    r"(?:l+i+s+t+e+n+i+n+g+\s+t+o+\s+|p+l+a+y+i+n+g+\s+|w+a+t+c+h+i+n+g+\s+|g+o+i+n+g+\s+t+o+\s+(?:t+h+e+\s+)?)?"
    r"(?:a+\s+l+o+t+\s+o+f+\s+|s+o+m+e+\s+)?"
    r"(?:m+u+s+i+c+|r+a+p+|h+i+p+h+o+p+|p+o+p+|r+o+c+k+|r+n+b+|k+p+o+p+|e+d+m+|j+a+z+z+|m+e+t+a+l+|i+n+d+i+e+|"
    r"s+i+n+g+i+n+g+|g+u+i+t+a+r+|p+i+a+n+o+|"
    r"g+a+m+e+s*|v+i+d+e+o+\s+g+a+m+e+s*|g+a+m+i+n+g+|v+a+l+o+r+a+n+t+|f+o+r+t+n+i+t+e+|m+i+n+e+c+r+a+f+t+|"
    r"r+o+b+l+o+x+|c+o+d+|w+a+r+z+o+n+e+|f+i+f+a+|g+t+a+|l+e+a+g+u+e+|p+u+b+g+|a+p+e+x+|p+s+5+|p+c+|"
    r"m+o+v+i+e+s*|f+i+l+m+s*|a+n+i+m+e+|m+a+n+g+a+|s+h+o+w+s*|s+e+r+i+e+s+|k+d+r+a+m+a+|h+o+r+r+o+r+|c+o+m+e+d+y+|a+c+t+i+o+n+|"
    r"s+p+o+r+t+s*|g+y+m+|w+o+r+k+i+n+g+\s+o+u+t+|w+o+r+k+o+u+t+s*|f+i+t+n+e+s+s+|f+o+o+t+b+a+l+l+|s+o+c+c+e+r+|"
    r"b+a+s+k+e+t+b+a+l+l+|t+e+n+n+i+s+|s+w+i+m+m+i+n+g+|r+u+n+n+i+n+g+|b+o+x+i+n+g+|"
    r"t+r+a+v+e+l+l*i+n+g*|t+r+a+v+e+l+|h+i+k+i+n+g+|c+a+m+p+i+n+g+|b+e+a+c+h+|"
    r"d+r+a+w+i+n+g+|a+r+t+|p+a+i+n+t+i+n+g+|p+h+o+t+o+g+r+a+p+h+y+|r+e+a+d+i+n+g+|b+o+o+k+s*|c+o+o+k+i+n+g+|"
    r"b+a+k+i+n+g+|d+a+n+c+i+n+g+|s+h+o+p+p+i+n+g+|f+a+s+h+i+o+n+|c+a+r+s+|b+i+k+e+s+|c+o+d+i+n+g+|d+o+g+s+|c+a+t+s+)"
    r"(?:\s+(?:m+u+s+i+c+|g+a+m+e+s*|m+o+v+i+e+s*)|\s+a+n+d+\s+(?:m+u+s+i+c+|g+a+m+i+n+g+|g+a+m+e+s*|m+o+v+i+e+s*|a+n+i+m+e+|g+y+m+|s+p+o+r+t+s*|t+r+a+v+e+l+l*i+n+g*|r+e+a+d+i+n+g+))?|"
    r"(?:m+y+\s+(?:f+a+v+o*u*r+i+t+e+\s+|f+a+v+\s+)?(?:h+o+b+b*y+|h+o+b+b*i+e+s+|s+p+o+r+t+|g+a+m+e+|m+u+s+i+c+|m+o+v+i+e+)\s+(?:i+s+|a+r+e+)\s+"
    r"(?:m+u+s+i+c+|r+a+p+|p+o+p+|g+a+m+i+n+g+|v+a+l+o+r+a+n+t+|m+i+n+e+c+r+a+f+t+|f+o+r+t+n+i+t+e+|a+n+i+m+e+|"
    r"f+o+o+t+b+a+l+l+|s+o+c+c+e+r+|b+a+s+k+e+t+b+a+l+l+|g+y+m+|t+r+a+v+e+l+l*i+n+g*|d+r+a+w+i+n+g+|r+e+a+d+i+n+g+|c+o+o+k+i+n+g+))|"
    r"(?:g+o+i+n+g+\s+t+o+\s+(?:t+h+e+\s+)?g+y+m+|w+o+r+k+i+n+g+\s+o+u+t+|p+l+a+y+i+n+g+\s+(?:f+o+o+t+b+a+l+l+|s+o+c+c+e+r+|b+a+s+k+e+t+b+a+l+l+|v+a+l+o+r+a+n+t+|f+o+r+t+n+i+t+e+|m+i+n+e+c+r+a+f+t+|g+t+a+|f+i+f+a+)|w+a+t+c+h+i+n+g+\s+a+n+i+m+e+)|"
    r"(?:r+a+p+|h+i+p+h+o+p+|p+o+p+\s+m+u+s+i+c+|r+o+c+k+\s+m+u+s+i+c+|k+p+o+p+|r+n+b+|e+d+m+|"
    r"v+a+l+o+r+a+n+t+|f+o+r+t+n+i+t+e+|m+i+n+e+c+r+a+f+t+|r+o+b+l+o+x+|f+i+f+a+|g+t+a+|"
    r"a+n+i+m+e+|m+a+n+g+a+|k+d+r+a+m+a+|h+o+r+r+o+r+\s+m+o+v+i+e+s*|"
    r"f+o+o+t+b+a+l+l+|s+o+c+c+e+r+|b+a+s+k+e+t+b+a+l+l+|t+e+n+n+i+s+|g+y+m+|f+i+t+n+e+s+s+|"
    r"t+r+a+v+e+l+l*i+n+g*|h+i+k+i+n+g+|d+r+a+w+i+n+g+|p+h+o+t+o+g+r+a+p+h+y+)"
    r")(?:(?:\s*,\s*|\s+)(?:u+|w+b+u+|h+b+u+)\??)?$",
    re.IGNORECASE,
)

ASK_HOBBIES_BASES = (
    # Music Questions
    "what music do you like", "what music do u like", "what music u like", "what kind of music do u like",
    "what kind of music u like", "what type of music do u like", "what music do u listen to",
    "what music u listen to", "what kind of music do u listen to", "whats ur favorite music",
    "whats ur fav song", "do u like music", "do u listen to music", "do u like rap", "do u like pop",
    # Gaming Questions
    "what games do you play", "what games do u play", "what games u play", "what video games do u play",
    "what games do u like", "whats ur favorite game", "whats ur fav game", "do u play games",
    "do u play video games", "are u a gamer", "do u play valorant", "do u play fortnite", "do u play minecraft",
    # Movies / Anime / Shows Questions
    "what movies do you like", "what movies do u like", "what movies u like", "what kind of movies do u like",
    "whats ur favorite movie", "whats ur fav movie", "what shows do u watch", "what series do u like",
    "do u watch anime", "do u like anime", "whats ur favorite anime", "do u like horror movies",
    # Sports / Gym Questions
    "what sports do you like", "what sports do u like", "what sports do u play", "do u play sports",
    "do u go to the gym", "do u workout", "do u like football", "do u like basketball",
    # General Hobbies / Free Time Questions
    "what are your hobbies", "what r ur hobbies", "what are ur hobbies", "whats your hobby",
    "whats ur hobby", "what are your interests", "what r ur interests", "what do you do for fun",
    "what do u do for fun", "what do u like to do", "what do u like doing", "what do u like doing for fun",
    "what do u do in ur free time", "any hobbies", "ur hobbies?", "hobbies?", "music?", "games?", "anime?",
)

CLEAN_TELL_TEMPLATES = (
    # Music Statements
    "i like music", "i love music", "i like pop music", "i like rap", "i like rap music",
    "i listen to rap", "i listen to pop", "i listen to rock", "i listen to rnb", "i listen to kpop",
    "i love hiphop", "i love rnb", "i like rock music", "i like edm", "i like indie music",
    "i like singing", "i play guitar", "i play piano", "music and gaming",
    # Gaming Statements
    "i like gaming", "i love gaming", "i like playing games", "i play video games",
    "i play valorant", "i play fortnite", "i play minecraft", "i play roblox", "i play cod",
    "i play fifa", "i play gta", "i play league", "i play on ps5", "i play on pc",
    # Movies / Anime Statements
    "i like movies", "i love movies", "i like watching movies", "i like horror movies",
    "i like comedy movies", "i like action movies", "i love anime", "i like anime",
    "i watch anime", "watching anime", "i like kdrama", "i like netflix", "anime and gaming",
    # Sports / Gym Statements
    "i like sports", "i love sports", "i like going to the gym", "going to the gym",
    "i love the gym", "working out", "i like working out", "i like fitness",
    "i play football", "i like football", "i play soccer", "i like soccer",
    "i play basketball", "i like basketball", "i like swimming", "i like running", "i like boxing",
    # General Hobbies / Travel / Art / Books
    "i like traveling", "i love traveling", "i like travelling", "i like hiking",
    "i like drawing", "i love drawing", "i like art", "i like painting", "i like photography",
    "i like reading", "i like reading books", "i love books", "i like cooking", "i love cooking",
    "i like dancing", "i like shopping", "i like cars", "i like coding",
    "my hobby is drawing", "my hobby is gaming", "my hobby is music", "my hobby is football",
)

DIRECT_HOBBY_ANSWERS = (
    "rap", "hiphop", "pop music", "rock music", "kpop", "rnb", "edm",
    "valorant", "fortnite", "minecraft", "roblox", "fifa", "gta",
    "anime", "manga", "kdrama", "horror movies",
    "football", "soccer", "basketball", "tennis", "gym", "fitness",
    "traveling", "travelling", "hiking", "drawing", "photography",
)


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    if valid.shape[0] == 0:
        return ""
    return valid.tobytes().decode("ascii", errors="ignore")


def _collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def _canonical_token_sig(s):
    has_q = "?" in s
    clean_w = "".join(ch if ("a" <= ch <= "z" or "0" <= ch <= "9" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, has_q)


def categorize_line_across_all_steps(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return 0, ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.")
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return 0, ""

    if STEP1_TO_9_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S10_VOCAB:
            return 0, ""

    if STRICT_ASK_HOBBIES_RE.match(s):
        return 25, s

    if len(words) == 1 and "?" in s:
        return 0, ""

    if STRICT_TELL_HOBBIES_RE.match(s):
        return 26, s

    return 0, ""


def _safe_fetch_parquet_hobbies_lines(ask_map, tell_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
    stats = {"scanned": 0, "blocked_overlaps": 0, "ask_added": 0, "tell_added": 0}
    try:
        from datasets import load_dataset
    except ImportError:
        return stats

    dialog_repos = (
        ("daily_dialog", "refs/convert/parquet", "dialog", 2500),
        ("bavard/personachat_truecased", "refs/convert/parquet", "history", 1500),
        ("AlekseyKorshuk/persona-chat", "refs/convert/parquet", "utterances", 1500),
    )

    for repo, rev, col, max_rows in dialog_repos:
        try:
            ds = load_dataset(repo, split="train", revision=rev, streaming=True)
            c = 0
            for row in ds:
                if c >= max_rows:
                    break
                c += 1
                stats["scanned"] += 1
                val = row.get(col, ())
                turns = (val,) if isinstance(val, str) else (val if isinstance(val, (list, tuple)) else ())
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    for sent in re.split(r"[.!]+", turn[:110]):
                        cid, clean_s = categorize_line_across_all_steps(sent)
                        if 1 <= cid <= 24:
                            stats["blocked_overlaps"] += 1
                        elif cid == 25:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in ask_map:
                                ask_map[sig] = clean_s
                                stats["ask_added"] += 1
                        elif cid == 26:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in tell_map:
                                tell_map[sig] = clean_s
                                stats["tell_added"] += 1
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_hobbies_files(
    ask_path="hobbies_ask.txt",
    tell_path="hobbies_tell.txt",
    neg_path="negatives.txt",
):
    ask_map = {}
    tell_map = {}

    for base in ASK_HOBBIES_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 25:
                ask_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "hey", "so", "btw", "wbu"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 25:
                    sig = _canonical_token_sig(c)
                    if sig not in ask_map:
                        ask_map[sig] = c

    for tmpl in CLEAN_TELL_TEMPLATES:
        for sfx in ("", " , u", " u?", " wbu?"):
            cand = f"{tmpl}{sfx}".strip()
            cid, c = categorize_line_across_all_steps(cand)
            if cid == 26:
                sig = _canonical_token_sig(c)
                if sig not in tell_map:
                    tell_map[sig] = c

    for d_ans in DIRECT_HOBBY_ANSWERS:
        for sfx in ("", " u?", " wbu?"):
            cand = f"{d_ans}{sfx}".strip()
            cid, c = categorize_line_across_all_steps(cand)
            if cid == 26:
                sig = _canonical_token_sig(c)
                if sig not in tell_map:
                    tell_map[sig] = c

    hf_stats = _safe_fetch_parquet_hobbies_lines(ask_map, tell_map)

    for k_sig in list(tell_map.keys()):
        if k_sig in ask_map:
            del tell_map[k_sig]

    n_ask, b_ask = _write_u8_file(ask_path, ask_map.values())
    n_tell, b_tell = _write_u8_file(tell_path, tell_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in list(ask_map.values()) + list(tell_map.values()):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[PARQUET HOBBY ANALYZER] Scanned {hf_stats['scanned']} rows | "
        f"Blocked {hf_stats['blocked_overlaps']} Step 1-9 overlaps | "
        f"Added {hf_stats['ask_added']} ask & {hf_stats['tell_added']} tell lines from Parquet!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 10 HOBBIES RULES 12+13] Saved {n_ask} unique ask lines to '{ask_path}' ({b_ask / 1024.0:.2f} KB) "
        f"& {n_tell} unique tell lines to '{tell_path}' ({b_tell / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_hobbies_files()