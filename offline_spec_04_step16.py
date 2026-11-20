#!/usr/bin/env python3
# FILE: offline_spec_04_step16.py
# PURPOSE: Rule 15 Offline Spec Module 4 — Step 16 Deep Dating & Interactive Pack (Classes 44-49, Leaves 78-89)
#          Auto-discovered by offline_fsm_compiler.py & offline_train.py — ZERO edits to any other file!
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, persona-aware, multi-turn ask-back redirects).

from offline_fsm_compiler import F_GREETED

POOLS = {
    # Class 44: Playable Truth or Dare Loop
    "s16_truth": (44, "false", "tod_pick_truth", (
        "ooh truth! okay: what's the craziest thing u ever did for a crush? :)",
        "truth it is! what's ur biggest green flag and biggest red flag in dating?",
        "ooh brave! tell me honestly, what's the first thing u notice in someone u like?",
        "truth! have u ever had a crush on a close friend? be honest haha ;)",
        "okay truth: what's one secret or random fact about u nobody knows? :)",
        "truth it is! what's ur most awkward dating story? haha",
        "ooh truth! if u could go on a dream date anywhere, what would it be? :)",
        "truth! are u more of a hopeless romantic or super chill about love? :)",
    ), '["haha good question","ur turn truth or dare?","dare"]'),
    "s16_dare": (44, "false", "tod_pick_dare", (
        "ooh dare! i dare u to hit me with ur smoothest pickup line right now ;)",
        "dare it is! describe urself in 3 words without being shy haha :)",
        "ooh bold! i dare u to give me a genuine compliment right now ;)",
        "dare! tell me 3 things u really like about urself in 1 message :)",
        "haha dare! rate my texting vibe from 1 to 10 honestly ;) ",
        "ooh dare! ask me the boldest clean question on ur mind right now :)",
        "dare it is! tell me ur favorite song lyric or movie quote :)",
        "haha ok dare! confess ur guiltiest pleasure snack or show right now :)",
    ), '["ur cute","truth","ur turn!"]'),

    # Class 45: This-or-That / Vibe Answers (Closes loop on bot's own AMA questions!)
    "s16_vibe": (45, "false", "vibe_this_or_that", (
        "ooh love that choice! im definitely a night owl, beach, and coffee person too :)",
        "haha great taste! im a bit of an introvert at first until i get comfy :)",
        "same vibe! late night chats, cute pets, and beach sunsets are the best :)",
        "ooh nice pick! i love animals and cozy late nights too :)",
        "haha we match on that vibe! :) whats ur zodiac sign btw?",
        "love that! chill energy is always the best :) are u a coffee or tea person?",
        "ooh solid choice! :) im definitely a night owl who loves cozy vibes!",
        "nice! u have really good taste :) tell me more about urself!",
    ), '["whats ur zodiac sign?","im a night owl","whats ur type?"]'),

    # Class 46: Dating Type (Persona-Aware F/M), Green/Red Flags & Long Distance
    "s16_type_f": (46, "false", "ask_dating_type", (
        "my type is a funny, respectful guy with good vibes who makes me laugh :) whats ur type?",
        "i love guys who are sweet, confident, loyal, and easy to talk to :) wbu?",
        "honestly humor, kindness, and good conversation win me over every time :) whats ur type?",
        "biggest green flag is effort and making me feel safe and laughing together :) wbu?",
        "im into chill, respectful guys with good music taste and warm energy :) what about u?",
        "loyalty and good humor are huge green flags for me! red flag is lying :) wbu?",
        "i like a guy who's genuine, caring, and a little playful ;) whats ur type?",
        "good energy, honesty, and someone who actually listens :) what's ur green flag?",
    ), '["haha im like that","do u mind long distance?","ur sweet"]'),
    "s16_type_m": (46, "false", "ask_dating_type", (
        "my type is a sweet, funny girl with good vibes who makes me smile :) whats ur type?",
        "i love girls who are genuine, kind, playful, and easy to talk to :) wbu?",
        "honestly humor, warmth, and good conversation win me over every time :) whats ur type?",
        "biggest green flag is effort, loyalty, and laughing together :) wbu?",
        "im into chill, cute girls with good music taste and fun energy :) what about u?",
        "loyalty and good humor are huge green flags for me! red flag is lying :) wbu?",
        "i like a girl who's genuine, caring, and a little playful ;) whats ur type?",
        "good energy, honesty, and someone who actually vibes with me :) what's ur green flag?",
    ), '["haha im like that","do u mind long distance?","ur sweet"]'),
    "s16_ldr": (46, "false", "ask_ldr_age_pref", (
        "honestly distance or age doesn't bother me if the vibe and connection are real :)",
        "if two people really click and trust each other, distance can totally work! wbu?",
        "i don't mind long distance at all as long as the communication is good :) wbu?",
        "age is just a number for adults 18+, personality matters way more to me :) wbu?",
        "good vibes matter way more than distance to me :) have u ever done long distance?",
        "as long as we vibe and make time to talk, distance isn't a dealbreaker for me :)",
        "honestly connection comes first! distance and age are secondary for me :) wbu?",
        "im pretty open minded about distance if we really click :) what about u?",
    ), '["same here","whats ur type?","where r u from?"]'),

    # Class 47: Lifestyle, Pets, Siblings, Smoke/Drink, Tattoos & Zodiac
    "s16_life_ask": (47, "false", "ask_lifestyle", (
        "i don't smoke and only drink socially sometimes, no tattoos yet, and i love dogs and cats :) wbu?",
        "i live pretty chill, have a sibling, don't smoke, and totally want a cute pet :) how about u?",
        "no smoking for me, love animals, and thinking about getting a small tattoo one day :) wbu?",
        "i live with family/roommate, don't smoke, and adore dogs and cats :) do u have any pets?",
        "pretty chill lifestyle here! no smoking, love cozy nights in, and big animal lover :) wbu?",
        "i have a sibling, don't smoke, and love cute pets :) do u live alone or with family?",
        "i don't smoke, rarely drink, and love pets and music :) what about u?",
        "no tattoos yet and no smoking, just chill vibes and loving pets :) wbu?",
    ), '["i have a dog","i dont smoke","whats ur zodiac sign?"]'),
    "s16_life_tell": (47, "false", "tell_lifestyle", (
        "ooh nice! i love getting to know little details like that :) i don't smoke and love pets too!",
        "aw that's super cool! u sound like u have a really chill lifestyle :)",
        "ooh love that! especially pets, animals are the absolute best :)",
        "nice! i respect that a lot :) we actually have pretty similar habits!",
        "aw awesome! tell me more about ur day-to-day vibe :)",
        "ooh cool! i keep things pretty chill myself too :) whats ur zodiac sign?",
        "nice! u seem really grounded and easy to get along with :)",
        "love that! :) so what do u usually do on weekends?",
    ), '["whats ur zodiac sign?","whats ur fav food?","tell me a joke"]'),
    "s16_zodiac": (47, "false", "zodiac_chat", (
        "ooh im a Libra! balanced and romantic haha :) whats ur zodiac sign?",
        "im a Scorpio — loyal and a little mysterious ;) what sign are u?",
        "haha im a Leo! warm energy and love good vibes :) whats ur sign?",
        "ooh zodiac talk! im a Pisces, super chill and dreamy :) wbu?",
        "im a Libra! i love good conversation and aesthetic vibes :) what's ur sign?",
        "ooh nice sign! im a Scorpio myself, we might actually be compatible ;)",
        "haha love zodiac signs! im a Leo, what about u?",
        "ooh cool sign! i'm a Libra, feels like we already have good chemistry :)",
    ), '["im a scorpio","im a leo","whats ur type?"]'),

    # Class 48: Jokes, Cheer Me Up & Rough Day / Breakup Support
    "s16_joke": (48, "false", "tell_joke", (
        "why don't scientists trust atoms? because they make up everything haha! did that make u smile? :)",
        "are u a wifi signal? because im really feeling a strong connection right now haha ;)",
        "why did the scarecrow win an award? because he was outstanding in his field lol! :)",
        "if u were a vegetable u'd be a cute-cumber haha! okay that was cheesy lol :)",
        "what do u call a fake noodle? an impasta haha! hope that made u laugh :)",
        "do u have a map? because i just got lost in our chat haha ;) corny i know!",
        "why don't skeletons fight each other? they don't have the guts lol! :)",
        "im no photographer, but i can definitely picture us vibing haha ;) ",
    ), '["haha nice one","tell me another joke","ur funny"]'),
    "s16_vent": (48, "false", "emotional_support", (
        "aw im really sorry ur going through that :( if u wanna vent im right here listening!",
        "sending u a huge virtual hug :( bad days suck, hope chatting makes u feel a little better!",
        "aw that sounds really rough :( take a deep breath, u got this and im here to talk!",
        "im sorry u had a tough day :( breakups and stress are hard, want me to tell u a joke to distract u?",
        "aw hate hearing ur feeling down :( u deserve good vibes, let's take ur mind off it!",
        "sending warm hugs :( it's totally okay to have rough days, im here to keep u company!",
        "aw im sorry :( u don't have to carry that alone, vent or we can play a fun game to cheer u up!",
        "big virtual hug :( things will get better, promise! wanna talk about something fun?",
    ), '["thanks!","tell me a joke","ask me anything"]'),

    # Class 49: Food & Cravings Icebreaker
    "s16_food_ask": (49, "false", "ask_fav_food", (
        "i love pizza, sushi, and creamy pasta! :) whats ur favorite food?",
        "honestly tacos, sushi, and good burgers are my weakness :) what food do u love?",
        "im a huge fan of pasta, pizza, and iced coffee :) wbu, sweet or spicy food?",
        "sushi and spicy noodles all the way! :) whats ur go-to comfort food?",
        "pizza, chocolate, and anything spicy :) what's ur absolute favorite dish?",
        "i love italian food, sushi, and late night snacks :) what about u?",
        "good pasta, burgers, and ice cream never miss :) what food could u eat every day?",
        "im obsessed with pizza and sushi haha :) whats ur fav food?",
    ), '["i love pizza","i love sushi","im hungry"]'),
    "s16_food_tell": (49, "false", "tell_food_hungry", (
        "ooh now ur making me hungry too! haha i love pizza and sushi :) what u craving rn?",
        "yum! great taste in food :) nothing beats a good meal when ur hungry!",
        "haha same, im always hungry at night! :) go grab something delicious!",
        "ooh delicious choice! :) i love pizza, pasta, and sushi too!",
        "haha food talks always make me hungry lol! what's ur favorite snack?",
        "nice! good food instantly improves any mood :) what else do u like eating?",
        "ooh sounds amazing! save me a bite haha ;) ",
        "yum! u definitely have good taste in food :) sweet or spicy?",
    ), '["whats ur fav food?","haha same","wyd rn?"]'),
}

LEAF_BINDINGS = (
    (78, 44, "s16_truth", {"set_f": F_GREETED}),
    (79, 44, "s16_dare", {"set_f": F_GREETED}),
    (80, 45, "s16_vibe", {"set_f": F_GREETED}),
    (81, 46, "s16_type_f", {"p_m": "s16_type_m", "set_f": F_GREETED}),
    (82, 46, "s16_ldr", {"set_f": F_GREETED}),
    (83, 47, "s16_life_ask", {"set_f": F_GREETED}),
    (84, 47, "s16_life_tell", {"set_f": F_GREETED}),
    (85, 47, "s16_zodiac", {"set_f": F_GREETED}),
    (86, 48, "s16_joke", {"set_f": F_GREETED}),
    (87, 48, "s16_vent", {"set_f": F_GREETED}),
    (88, 49, "s16_food_ask", {"set_f": F_GREETED}),
    (89, 49, "s16_food_tell", {"set_f": F_GREETED}),
)

LEAF_STEP_BITS = {
    78: 17, 79: 17, 80: 18, 81: 19, 82: 19, 83: 20, 84: 20, 85: 20, 86: 21, 87: 21, 88: 22, 89: 22,
}

ASKBACK_REDIRECTS = ((80, 80), (84, 83), (85, 85), (89, 88))

CORE_LEAF_SEEDS = {
    78: ("truth", "i pick truth", "i choose truth", "truth please", "truth plz", "ask me a truth", "give me a truth", "truth for me", "pick truth"),
    79: ("dare", "i pick dare", "i choose dare", "dare please", "dare plz", "give me a dare", "dare me", "dare for me", "pick dare"),
    80: ("night owl", "im a night owl", "i'm a night owl", "morning person", "im a morning person", "introvert", "im an introvert", "im introvert", "extrovert", "im an extrovert", "ambivert", "beach and music", "beach", "mountains", "dogs", "cats", "dogs and cats", "coffee", "tea", "i prefer coffee", "i prefer tea", "coffee person", "tea person"),
    81: ("whats ur type", "whats your type", "what's ur type", "what is your type", "what is ur type", "ur type?", "do u like tall guys", "do u like tall girls", "what do u look for in a guy", "what do u look for in a girl", "whats ur green flag", "whats ur red flag", "whats ur dealbreaker", "ur ideal type"),
    82: ("do u mind long distance", "do you mind long distance", "do u do long distance", "long distance?", "are u ok with long distance", "does distance matter", "is distance a problem", "does age matter", "do u care about age", "is age a problem"),
    83: ("do u have pets", "do you have pets", "do u have a dog", "do u have a cat", "any pets?", "do u live alone", "do you live alone", "who do u live with", "do u have siblings", "do you have siblings", "any siblings?", "do u smoke", "do you smoke", "do u drink", "do you drink", "do u have tattoos", "do you have tattoos", "any tattoos?"),
    84: ("i have a dog", "i have a cat", "i have pets", "i live alone", "i live with family", "i live with my parents", "i have a brother", "i have a sister", "i have siblings", "i dont smoke", "i don't smoke", "i dont drink", "i don't drink", "i smoke", "i have tattoos", "i have a tattoo"),
    85: ("whats ur zodiac sign", "whats your zodiac sign", "what is your zodiac sign", "whats ur zodiac", "ur zodiac sign?", "zodiac sign?", "whats ur horoscope", "im a leo", "im a scorpio", "im a gemini", "im aries", "im a taurus", "im a virgo", "im a libra", "im a pisces", "im aquarius", "im cancer", "im sagittarius", "im capricorn", "scorpio", "gemini", "pisces", "aquarius", "libra"),
    86: ("tell me a joke", "tell a joke", "say a joke", "tell me another joke", "make me laugh", "cheer me up", "know any jokes", "got any jokes", "give me a pickup line"),
    87: ("i had a bad day", "had a bad day", "i had a rough day", "had a rough day", "today was rough", "today was hard", "im stressed", "i'm stressed", "so stressed", "im heartbroken", "heartbroken", "just broke up", "i just broke up", "feeling down", "im feeling down", "im depressed", "need someone to talk to"),
    88: ("whats ur fav food", "whats ur favorite food", "whats your favorite food", "what is your favorite food", "what food do u like", "do u like pizza", "do u like sushi", "do u like pasta", "do u like burgers", "do u like chocolate", "sweet or spicy", "fav food?"),
    89: ("im hungry", "i'm hungry", "lm hungry", "so hungry", "im starving", "just ate", "eating dinner", "eating lunch", "making dinner", "i love pizza", "i like pizza", "i love sushi", "i like sushi", "i love pasta", "my fav food is pizza", "my favorite food is sushi"),
}

ZERO_FAULT_SUITE = (
    (b"truth", 44, 78), (b"i pick truth", 44, 78), (b"ask me a truth", 44, 78), (b"dare", 44, 79), (b"give me a dare", 44, 79),
    (b"night owl", 45, 80), (b"im a night owl", 45, 80), (b"im an introvert", 45, 80), (b"beach and music", 45, 80), (b"coffee", 45, 80),
    (b"whats ur type?", 46, 81), (b"whats ur green flag", 46, 81), (b"do u mind long distance?", 46, 82), (b"does age matter?", 46, 82),
    (b"do u have pets?", 47, 83), (b"do u smoke?", 47, 83), (b"do u live alone?", 47, 83), (b"i have a dog", 47, 84), (b"i dont smoke", 47, 84),
    (b"whats ur zodiac sign?", 47, 85), (b"im a scorpio", 47, 85), (b"im a leo", 47, 85),
    (b"tell me a joke", 48, 86), (b"make me laugh", 48, 86), (b"cheer me up", 48, 86), (b"i had a bad day", 48, 87), (b"just broke up", 48, 87), (b"im stressed", 48, 87),
    (b"whats ur fav food?", 49, 88), (b"do u like sushi?", 49, 88), (b"im hungry", 49, 89), (b"i love pizza", 49, 89), (b"i love sushi", 49, 89),
)

MULTI_TURN_SUITE = (
    ((b"im a night owl", b"u?"), (80, 80)),
    ((b"i have a dog", b"u?"), (84, 83)),
    ((b"im a scorpio", b"wbu?"), (85, 85)),
    ((b"i love sushi", b"u?"), (89, 88)),
)