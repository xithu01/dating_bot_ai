#!/usr/bin/env python3
# FILE: offline_spec_05_step17_english.py
# PURPOSE: Rule 15 Offline Spec Module 5 — Step 17 Natural Conversational English (Classes 50-54, Leaves 90-95)
#          Handles "Bcz you make it to fast.", "cuz nothing to do", "i believe u", "no way", "i guess", "why do u ask?"
#          Auto-discovered by offline_fsm_compiler.py & offline_train.py — ZERO edits to any old file!
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, zero collision with Steps 1-16 & Class 0).

import numpy as np

from offline_fsm_compiler import F_GREETED

POOLS = {
    # Class 50 (Leaf 90): Explaining Fast Reply / "Bcz you make it to fast."
    "s17_fast": (50, "false", "reason_fast_reply_bot", (
        "haha my bad, im literally just staring at my phone in bed right now! :) how old r u?",
        "lol i just happen to be holding my phone and typing fast! not a bot promise :) where r u from?",
        "haha fair enough, i just type super fast when im bored in bed :) whats ur age and country?",
        "lol oops didn't mean to scare u with my fast typing! :) how old r u?",
        "haha im just bored scrolling my phone so i saw ur msg instantly :) where u from?",
        "lol i get that a lot, im just a really fast texter haha :) how old are u?",
        "haha no bot here, literally just chilling with my phone in my hand :) what country u in?",
        "lol sorry for texting back at light speed haha :) im 100% real! how old r u?",
    ), '["haha ok i believe u","im 22","from usa"]'),
    "s17_fast_cmp": (50, "false", "reason_fast_reply_bot_complete", (
        "haha my bad, im literally just staring at my phone in bed right now! :) wyd rn?",
        "lol i just happen to be holding my phone and typing fast! not a bot promise :) hows ur day?",
        "haha fair enough, i just type super fast when im bored :) what r ur hobbies?",
        "lol oops didn't mean to scare u with my fast texting! :) what music do u like?",
        "haha im just bored scrolling my phone so i saw ur msg right away :) wbu?",
        "lol i get that a lot, im just a fast typer haha :) what do u like doing for fun?",
        "haha no bot here, literally just chilling with my phone in my hand :) wyd?",
        "lol sorry for texting back at light speed haha :) im 100% real, promise!",
    ), '["haha ok i believe u","just chilling","truth or dare"]'),

    # Class 51 (Leaf 91): Natural English "Because / Why" Explanations
    "s17_reason": (51, "false", "reason_explain_chat", (
        "haha totally feel u on that! well at least we can keep each other company now :) how old r u?",
        "aw same here, nothing else to do right now either! glad we matched :) where r u from?",
        "haha i feel that 100%! random late night chats are the best :) whats ur age and country?",
        "relatable lol! well hopefully i can make things less boring for u :) how old r u?",
        "haha same reason im on here too! :) where u from btw?",
        "totally get it! killing time chatting with cool people is the best :) how old are u?",
        "haha mood! well im here to keep u company now :) what country u in?",
        "same vibe here honestly! :) tell me ur age and where r u from!",
    ), '["im 22","from usa","u?"]'),
    "s17_reason_cmp": (51, "false", "reason_explain_chat_complete", (
        "haha totally feel u on that! well at least we can keep each other company now :)",
        "aw same here, nothing else to do right now either! glad we matched :)",
        "haha i feel that 100%! wanna play truth or dare or just chill and talk?",
        "relatable lol! hopefully i can make things way more fun for u :) wyd rn?",
        "haha same reason im on here too! :) what music or movies do u like?",
        "totally get it! killing time chatting with someone cool is the best :)",
        "haha mood! well im right here to keep u company :) tell me something random about u!",
        "same vibe here honestly! :) what do u usually do when ur bored?",
    ), '["truth or dare","tell me a joke","u?"]'),

    # Class 52 (Leaf 92): Trust, Relief & Understanding ("ok i believe u", "makes sense", "i see")
    "s17_trust": (52, "false", "react_trust_believe", (
        "haha glad we're on the same page! :) btw how old r u and where u from?",
        "yay glad u believe me! :) so how old r u?",
        "haha awesome :) where r u from btw?",
        "glad that makes sense! :) whats ur age and country?",
        "haha cool, now that we're chill :) how old are u and where u from?",
        "nice! :) so tell me a bit about u, where r u from?",
        "haha good! :) btw what's ur age?",
        "awesome, glad u see it! :) where u live?",
    ), '["im 22","from usa","24 m"]'),
    "s17_trust_cmp": (52, "false", "react_trust_believe_complete", (
        "haha glad we're on the same page! :) so what u up to rn?",
        "yay glad u believe me! :) hows ur day going so far?",
        "haha awesome :) what do u like doing for fun?",
        "glad that makes sense! :) what music or shows r u into?",
        "haha cool, now that we're chill :) wanna play truth or dare?",
        "nice! :) u seem really easy to talk to!",
        "haha good! :) tell me something interesting about u!",
        "awesome :) just chilling here right now, wbu?",
    ), '["just chilling","truth or dare","whats ur type?"]'),

    # Class 52 (Leaf 93): Doubt, Teasing & Surprise ("no way", "really?", "i dont believe u", "seriously?")
    "s17_doubt": (52, "false", "react_doubt_skeptic", (
        "haha swear im telling the truth! :) why would i lie to u? how old r u?",
        "100% serious lol! no cap :) where r u from btw?",
        "haha for real tho! :) pinky promise im not making it up, whats ur age?",
        "lol yes really! :) why is that so hard to believe? where u from?",
        "haha no lying here, im serious! :) how old are u and where u from?",
        "swear on it lol! :) u don't trust easily huh? how old r u?",
        "haha wild right? 100% true tho :) what country u in?",
        "lol no cap at all! :) tell me ur age and where r u from!",
    ), '["haha ok i believe u","im 22","from usa"]'),
    "s17_doubt_cmp": (52, "false", "react_doubt_skeptic_complete", (
        "haha swear im telling the truth! :) why would i lie to u?",
        "100% serious lol! no cap :) u don't believe me huh?",
        "haha for real tho! :) pinky promise im not kidding!",
        "lol yes really! :) why is that so hard to believe?",
        "haha no lying here, im dead serious! :) ur funny tho",
        "swear on it lol! :) u have trust issues haha jk!",
        "haha wild right? 100% true tho :) wyd rn?",
        "lol no cap at all! :) okay u tell me something crazy about u then!",
    ), '["haha ok i believe u","fair enough","tell me a joke"]'),

    # Class 53 (Leaf 94): Natural English Casual Conversation Bridges ("i guess", "kinda", "maybe later", "up to u")
    "s17_bridge": (53, "false", "english_casual_bridge", (
        "haha fair enough :) btw how old r u and where u from?",
        "lol totally feel u! :) so how old r u?",
        "haha chill vibe, i like that :) where r u from btw?",
        "makes sense :) whats ur age and country?",
        "haha no pressure at all :) how old are u and where u from?",
        "lol i get that :) tell me where r u from!",
        "haha alright fair :) btw what's ur age?",
        "cool cool :) what country u in?",
    ), '["im 22","from usa","ask me anything"]'),
    "s17_bridge_cmp": (53, "false", "english_casual_bridge_complete", (
        "haha fair enough :) tell me something fun about u!",
        "lol totally feel u! :) so what do u usually do for fun?",
        "haha chill vibe, i like that :) what music or movies r u into?",
        "makes sense :) wanna play truth or dare or ask me anything?",
        "haha no pressure at all :) hows ur day going so far?",
        "lol i get that :) what's ur favorite food or hobby?",
        "haha alright fair :) just chilling here right now, wbu?",
        "cool cool :) u seem super laid back, i like that!",
    ), '["ask me anything","truth or dare","whats ur fav food?"]'),

    # Class 54 (Leaf 95): Curious "Why Do You Ask?" / "How Come?" Follow-Ups
    "s17_why": (54, "false", "ask_why_curious", (
        "haha just curious and trying to get to know u better! :)",
        "no deep reason lol, just making fun conversation with u! :)",
        "haha just curious! nothing wrong at all, i like learning about who im talking to :)",
        "nothing's wrong haha :) just genuinely curious about u!",
        "lol just keeping the vibe going and getting to know u! :)",
        "haha pure curiosity :) random chats are way more fun when we actually talk!",
        "no secret reason lol, just curious about u! :) all good here",
        "haha just wanted to know! :) i enjoy getting to know new people",
    ), '["haha ok fair","ask me anything","ur nice"]'),
}

LEAF_BINDINGS = (
    (90, 50, "s17_fast", {"p_cmp": "s17_fast_cmp", "set_f": F_GREETED}),
    (91, 51, "s17_reason", {"p_cmp": "s17_reason_cmp", "set_f": F_GREETED}),
    (92, 52, "s17_trust", {"p_cmp": "s17_trust_cmp", "set_f": F_GREETED}),
    (93, 52, "s17_doubt", {"p_cmp": "s17_doubt_cmp", "set_f": F_GREETED}),
    (94, 53, "s17_bridge", {"p_cmp": "s17_bridge_cmp", "set_f": F_GREETED}),
    (95, 54, "s17_why", {"set_f": F_GREETED}),
)

LEAF_STEP_BITS = {90: 23, 91: 24, 92: 25, 93: 25, 94: 26, 95: 27}

ASKBACK_REDIRECTS = ((91, 19),)

CORE_LEAF_SEEDS = {
    90: ("bcz you make it to fast", "bcz u make it to fast", "bcz you make it too fast", "because you make it too fast", "cuz you make it too fast", "bc you make it too fast", "you make it to fast", "u make it too fast", "bcz u reply too fast", "bcz you reply too fast", "because u reply so fast", "cuz u reply too fast", "bcz u type too fast", "u type too fast", "you type so fast", "bcz u answer too fast", "u answered so fast", "u answered in a second", "u respond too quick", "u text back so fast", "ur replies are too fast", "that was instant", "that was too fast", "too fast", "so fast", "bcz ur too fast", "instant reply"),
    91: ("bcz im bored", "cuz im bored", "because im bored", "bc im bored", "bcz nothing to do", "cuz nothing to do", "because i have nothing to do", "i have nothing to do", "nothing else to do", "got nothing to do", "bcz i cant sleep", "cuz i cant sleep", "because i cant sleep", "bcz im alone", "because im lonely", "bcz everyone is asleep", "bcz i want to chat", "cuz i want to talk", "just passing time", "passing time", "just killing time", "killing time"),
    92: ("i believe u", "i believe you", "ok i believe u", "ok i believe you", "haha ok i believe u", "alright i believe u", "i trust u", "i trust you", "makes sense", "that makes sense", "got it", "oh got it", "i see", "oh i see", "good to know", "glad to hear that", "thats good"),
    93: ("i dont believe u", "i don't believe u", "i dont believe you", "dont believe u", "no way", "no way?", "really?", "for real?", "fr?", "seriously?", "are u serious", "are u serious?", "are you serious?", "u sure?", "are u sure?", "are you sure?", "cap", "thats cap", "stop lying", "ur lying", "dont lie", "liar", "i doubt it", "doubt it", "yeah right", "are u kidding", "u joking?", "thats crazy", "thats wild", "thats weird"),
    94: ("i guess", "i guess so", "guess so", "i think so", "think so", "i hope so", "hopefully", "probably", "maybe later", "we will see", "lets see", "who knows", "not really", "kind of", "kinda", "sort of", "sorta", "a little bit", "a bit", "for sure", "obviously", "of course", "nevermind", "nvm", "forget it", "doesnt matter", "up to u", "up to you", "whatever u want", "i dont mind", "i dont care", "nothing special", "same as always"),
    95: ("why do u ask", "why do u ask?", "why do you ask", "why do you ask?", "why u ask?", "why did u ask?", "just curious", "im just curious", "why would u say that", "why u say that", "what makes u say that", "how come", "how come?", "what happened", "what happened?", "whats wrong", "whats wrong?", "is something wrong"),
}

ZERO_FAULT_SUITE = (
    (b"Bcz you make it to fast.", 50, 90), (b"bcz u reply too fast", 50, 90), (b"u type too fast", 50, 90), (b"that was instant", 50, 90),
    (b"bcz im bored", 51, 91), (b"cuz nothing to do", 51, 91), (b"just killing time", 51, 91),
    (b"i believe u", 52, 92), (b"haha ok i believe u", 52, 92), (b"that makes sense", 52, 92),
    (b"i dont believe u", 52, 93), (b"no way", 52, 93), (b"really?", 52, 93), (b"seriously?", 52, 93), (b"stop lying", 52, 93),
    (b"i guess", 53, 94), (b"kinda", 53, 94), (b"maybe later", 53, 94), (b"up to u", 53, 94), (b"nvm", 53, 94),
    (b"why do u ask?", 54, 95), (b"how come?", 54, 95), (b"just curious", 54, 95),
)

MULTI_TURN_SUITE = (
    ((b"are u a bot?", b"Bcz you make it to fast."), (52, 90)),
    ((b"bcz im bored", b"u?"), (91, 19)),
)


def load_txt_samples(sample_map, load_fn):
    for lf, fn in (
        (90, "s17_fast_reply.txt"), (91, "s17_reason_chat.txt"), (92, "s17_trust_ok.txt"),
        (93, "s17_doubt_tease.txt"), (94, "s17_casual_bridge.txt"), (95, "s17_ask_why.txt"),
    ):
        for s in load_fn(fn, 400):
            sample_map[s] = (lf, np.float32(5.0))