# FILE: build_gender_txt.py
# PURPOSE: Generate Step 4 Gender and Age+Gender datasets cleanly (Classes 9..14 + Underage Traps)

import os

# CLASS 9: ASK GENDER CHOICE
ASK_CHOICE = (
    "m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m", "m f", "f m", "m f?", "f m?",
    "girl or boy", "girl or boy?", "boy or girl", "boy or girl?", "girl or male", "boy or female",
    "male or female", "male or female?", "female or male", "female or male?", "male or girl",
    "female or boy", "m or female", "f or male", "male or f", "female or m", "m or girl",
    "f or boy", "boy or m", "girl or f", "woman or man", "man or woman", "girl or woman",
    "boy or man", "female or girl", "male or boy", "woman or girl", "man or boy",
    "what gender are you", "what is your gender", "whats your gender", "what's your gender",
    "your gender", "ur gender", "u gender", "gender", "gender?", "your sex", "what sex are you",
    "are you male or female", "are you female or male", "are you a male or female",
    "are you a female or male", "which gender are you", "which sex are you",
    "what are you male or female", "tell me your gender", "r u m or f", "r u f or m",
)

# CLASS 10: ASK IS FEMALE (Binary Yes/No)
ASK_IS_FEMALE = (
    "f?", "girl?", "female?", "woman?", "w?", "are you a girl", "are you a girl?", "are you girl",
    "are u a girl", "are u girl", "you a girl?", "r u a girl", "r u girl", "you a girl", "u a girl",
    "ur a girl", "your a girl", "are you female", "are u female", "r u female", "you female",
    "u female", "are you a female", "are u a female", "r u a female", "r you a girl", "r you girl",
    "r you female", "you're a girl", "you are a girl", "you're female", "you are female",
    "is your gender female", "are you feminine", "are you a woman", "are u a woman", "r u a woman",
    "you a woman", "u a woman", "are u a gril?", "r u famle?", "ur a gurl?", "femael?",
)

# CLASS 11: ASK IS MALE (Binary Yes/No)
ASK_IS_MALE = (
    "m?", "boy?", "male?", "man?", "are you a boy", "are you a boy?", "are you boy", "are u a boy",
    "are u boy", "r u a boy", "r u boy", "you a boy", "u a boy", "ur a boy", "your a boy",
    "are you male", "are u male", "r u male", "you male", "u male", "are you a male", "are u a male",
    "r u a male", "r you a boy", "r you boy", "r you male", "you're a boy", "you are a boy",
    "you're male", "you are male", "is your gender male", "are you masculine", "are you a man",
    "are u a man", "r u a man", "you a man", "u a man", "are u a boi?", "r u maale?", "ur a boy?", "mael?",
)

# CLASS 12: STATE FEMALE (Statements only, no question marks)
STATE_FEMALE = (
    "f", "female", "famle", "famel", "femael", "femle", "girl", "gril", "gurl", "woman", "womman",
    "w", "f here", "female here", "girl here", "im a girl", "i'm a girl", "i am a girl",
    "im female", "i'm female", "im famle", "im a woman", "i'm a woman", "im girl",
)

# CLASS 13: STATE MALE (Statements only, no question marks)
STATE_MALE = (
    "m", "male", "maale", "mael", "boy", "boi", "man", "m here", "male here", "boy here",
    "im a boy", "i'm a boy", "i am a boy", "im male", "i'm male", "im a man", "i'm a man", "im boy",
)

# GENDERS FOR AGE COMBINATIONS
GENDER_ALIASES = ("f", "m", "female", "male", "famle", "girl", "boy", "woman", "man", "w")


def write_file(filename, lines_set):
    # Ensure lowercased and sorted output
    sorted_lines = sorted({s.lower().strip() for s in lines_set if s.strip()})
    with open(filename, "w", encoding="ascii") as f:
        for line in sorted_lines:
            f.write(line + "\n")
    print(f"[STEP 4] Wrote {len(sorted_lines)} lines to '{filename}'")


def generate_age_gender_files():
    valid_age_gender = set()
    underage_gender = set()

    # Formats: {age}{g}, {age} {g}, {g}{age}, {g} {age}, im {age} {g}, i am {age} {g}
    for age in range(10, 81):
        for g in GENDER_ALIASES:
            combos = [
                f"{age}{g}", f"{age} {g}", f"{g}{age}", f"{g} {age}",
                f"im {age} {g}", f"i am {age} {g}", f"im a {age} {g}"
            ]
            if age >= 18:
                valid_age_gender.update(combos)
            else:
                underage_gender.update(combos)

    write_file("age_gender.txt", valid_age_gender)
    write_file("underage_traps.txt", underage_gender)


def generate_all_gender_files():
    write_file("gender_ask_choice.txt", ASK_CHOICE)
    write_file("gender_ask_is_female.txt", ASK_IS_FEMALE)
    write_file("gender_ask_is_male.txt", ASK_IS_MALE)
    write_file("gender_state_female.txt", STATE_FEMALE)
    write_file("gender_state_male.txt", STATE_MALE)
    generate_age_gender_files()


if __name__ == "__main__":
    generate_all_gender_files()