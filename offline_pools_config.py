
#!/usr/bin/env python3
# FILE: offline_pools_config.py
# PURPOSE: Offline-Only Step 1-15 (Classes 0-43 + 99 -> 78 Leaves) FSM Table Compiler,
#          Context-Aware Ask-Back Redirect Table (`askback_redirect_u8`), HTTP Blob Packer & Zero-Fault Suite
# RULES: Enforces Rules 1-14 (Zero float64/int64, zero if-chains in production).

import numpy as np

NUM_LEAVES = 78  # 0..75 (Steps 1-15 + Reset), 76=context_ask_back ("u"/"u?"/"wbu?"), 77=status_chatting_with_u

# --- 64-BIT CONVERSATION FLAGS SPLIT INTO TWO np.uint32 BITMASKS (ZERO int64/uint64!) ---
F_GREETED = np.uint32(1 << 0)
F_ASKED_AGE = np.uint32(1 << 1)
F_ASKED_COUNTRY = np.uint32(1 << 2)
F_ASKED_HRU = np.uint32(1 << 3)
F_ASKED_GENDER = np.uint32(1 << 4)
F_ASKED_WYD = np.uint32(1 << 5)
F_TOLD_STATUS = np.uint32(1 << 6)
F_FLIRTED = np.uint32(1 << 7)
F_ASKED_LOOKING = np.uint32(1 << 8)
F_STATED_LOOKING = np.uint32(1 << 9)
F_BOT_NAMED = np.uint32(1 << 10)
F_USER_NAMED = np.uint32(1 << 11)
F_ASKED_REL = np.uint32(1 << 12)
F_ASKED_PRIV = np.uint32(1 << 13)
F_ASKED_PIC = np.uint32(1 << 14)
F_BOT_CHECKED = np.uint32(1 << 15)
F_ZERO = np.uint32(0)

RULE_STD = 0
RULE_AGE = 1
RULE_COUNTRY = 2
RULE_SET_G_F = 3
RULE_SET_G_M = 4
RULE_GREET = 5


def _pack_http(class_id, is_greeting_str, intent_str, reply_str, kbd_str):
    body = (
        f'{{"class_id":{class_id},"is_greeting":{is_greeting_str},'
        f'"intent":"{intent_str}","reply":"{reply_str}","kbd":{kbd_str}}}'
    ).encode("utf-8")
    hdr = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
    )
    return hdr + body


def _pool_specs():
    return {
        # --- CLASS 0, 7, 99 ---
        "reject": (0, "false", "not_valid", ("say hi, how r u, M or F, ur age (18+), or where u are from!",), '["hey","how r u?","im 22"]'),
        "underage": (7, "false", "underage_reject", ("sorry, 18+ adults only!",), '["Exit"]'),
        "reset": (99, "false", "system_reset", ("new chat started! say hi :)",), '["hey","hi!"]'),

        # --- STEP 1: GREETINGS ---
        "greet_first": (1, "true", "english_greeting", (
            "hey", "hi", "heyy", "yo", "hey there", "hii", "sup", "hello!",
            "hey how r u", "hi! hru?", "yo whats up", "heyy :)", "hiya", "hey! hows it going",
            "hiii", "sup! hru", "hello there", "hey u", "hi there!", "yo! how are u",
            "heyyy", "whats up!", "hey! nice to meet u", "hi :) how r u", "howdy!", "hey whats up",
            "hii there", "hello :)", "yo! whats good", "hey! hru doing", "hi! hows u", "heyy! whats up",
        ), '["good u?","how r u?"]'),
        "greet_rep": (1, "true", "english_greeting_repeat", (
            "hey again lol", "haha hi again!", "we already said hi :)", "yo again! how r u?",
            "hii again haha", "already said hey lol", "haha hey x2! hru?", "hi again! whats up?",
        ), '["haha im good","how r u?"]'),

        # --- STEP 2: AGE & COUNTRY ---
        "ask_age": (2, "false", "ask_age", (
            "22 u?", "im 23, how old r u?", "22! wbu?", "im 23 :)",
            "22 here, u?", "23! how old are u", "im 22, ur age?", "23 u?",
        ), '["im 21","im 24","im 28"]'),
        "ask_age_rep": (2, "false", "ask_age_repeat", (
            "im 22! where r u from?", "im 23 :) hows ur day going?", "22 here! what country u from?", "im 22 haha :) wyd rn?",
            "23 here! how r u doing?", "im 22 :) what u up to today?", "22 here! where u live?", "im 23 :) whats up?",
        ), '["from usa","im good"]'),
        "tell_age": (3, "false", "tell_age", (
            "nice! im 22, where u from?", "cool im 23! u from?", "nice age! im 22, from?", "22 here! where r u from?",
            "ooh nice, im 23! country?", "cool! 22 here, u from?", "nice! im 23, where u live?", "sweet, im 22! where u from?",
        ), '["usa","uk","tunisia"]'),
        "tell_age_cmp": (3, "false", "tell_age_complete", (
            "nice! im 22 too :) hows ur day going?", "cool age! im 23, what u up to today?", "sweet! 22 here :) how r u doing?", "ooh nice! im 23, hows it going?",
            "nice! 22 here, what u doing rn?", "cool! im 23 :) having a good day?", "awesome! im 22, whats up?", "nice age! 23 here, hows ur day?",
        ), '["going great!","chilling u?"]'),
        "tell_age_rep": (3, "false", "tell_age_repeat", (
            "haha u already said ur age! where r u from?", "ik ur age lol! what country u in?", "u told me ur age already haha! u from?", "haha yeah ik! where u from?",
            "already got ur age lol :) where u live?", "haha u said that! what country?", "ik lol! where r u from?", "yeah u told me haha! u from?",
        ), '["usa","uk","tunisia"]'),
        "tell_age_chg": (3, "false", "tell_age_changed", (
            "wait u said a different age before lol! where u from?", "haha changing ur age now? lol how r u", "wait didnt u say another age earlier lol?", "haha u changed ur age! which one is real?",
            "hold on u said a different age before lol!", "haha new age unlocked lol! whats up?", "wait u changed ur age haha!", "lol u said another age before! how r u?",
        ), '["haha jk","lol my bad"]'),
        "ask_ctry": (4, "false", "ask_country", (
            "from usa, u?", "uk! where u from?", "im from usa :)", "canada, wbu?",
            "usa here! u from?", "from uk, where r u from?", "usa! what about u?", "im in usa, u?",
        ), '["nice!","me too","cool"]'),
        "ask_ctry_rep": (4, "false", "ask_country_repeat", (
            "im from usa :) how old r u?", "im from uk! hows ur day going?", "usa here :) what u up to rn?", "from usa :) how r u doing?",
            "im from usa! what r ur hobbies?", "usa here :) having a good day?", "im from uk :) whats up?", "from usa :) wyd rn?",
        ), '["nice!","im good"]'),
        "tell_ctry": (5, "false", "tell_country", (
            "ooh nice! im from usa, how old r u?", "cool place! im from uk, age?", "nice! im from usa :) ur age?", "love that! im from usa, how old r u?",
            "awesome! usa here, how old u?", "nice country! im from uk, age?", "cool! im from canada, how old r u?", "sweet! usa here, ur age?",
        ), '["im 21","im 24","im 28"]'),
        "tell_ctry_cmp": (5, "false", "tell_country_complete", (
            "ooh nice! im from usa :) hows ur day going?", "cool place! im from uk, what u up to?", "love that! usa here :) how r u doing?", "awesome place! im from canada, whats up?",
            "nice! im from usa, what u doing rn?", "sweet! uk here :) having a good day?", "ooh cool! im from usa, hows it going?", "nice country! usa here, hows ur day?",
        ), '["going great!","chilling u?"]'),
        "tell_ctry_rep": (5, "false", "tell_country_repeat", (
            "ik ur from there lol! how old r u?", "haha u already told me ur country! whats up?", "yeah u said that already lol! how r u?", "ik lol! nice place :) what u up to?",
            "haha already got ur country! how old r u?", "u told me already lol! hows it going?", "yeah ik haha! whats up?", "haha u said that! how r u doing?",
        ), '["haha im good","chilling"]'),
        "tell_ctry_chg": (5, "false", "tell_country_changed", (
            "wait u said a different country before lol!", "haha moving countries fast lol! how r u?", "wait didnt u say another country earlier lol?", "haha traveling the world? u said another place before!",
            "hold on u changed ur country lol!", "haha new country unlocked! whats up?", "wait u said somewhere else before haha!", "lol changing countries now? how r u?",
        ), '["haha jk","traveling lol"]'),
        "tell_age_ctry": (6, "true", "tell_age_and_country", (
            "hey! nice im 22 from usa, how r u?", "hi! cool im 23 from uk, hru?", "nice to meet u! 22 from usa here :)", "heyy! im 22 from canada, hows it going?",
            "yo! 23 from usa here, whats up?", "hey there! im 22 from uk, how r u?", "hi! 22 from usa here, nice to meet u!", "heyy! cool, im 23 from usa, hru?",
        ), '["im good u?","doing great!"]'),

        # --- STEP 3: ASK HRU ---
        "ask_hru": (8, "false", "ask_how_are_you", (
            "im doing great! how old r u and where u from?", "doing good thanks! wbu, what's ur age and country?",
            "im good! im 22 from usa, how old r u and where u from?", "all good here! how old are u and where r u from?",
            "pretty good! im 23 from uk, ur age and country?", "doing great :) how old r u and where u live?",
            "im great thanks! wbu, age and where u from?", "good! chilling here, how old r u and what country?",
        ), '["im 22","from usa","24 from uk"]'),

        # --- STEP 4: GENDER (Persona-Aware Bot F vs Bot M + Smart Follow-Up Answer Without Re-Asking "u?") ---
        "g_choice_f": (9, "false", "ask_gender_choice", (
            "im F :) u?", "im a girl! wbu?", "female here, u?", "F! what about u?",
            "im a girl :) u M or F?", "F here! wbu?", "im female, how about u?", "girl here :) u?",
        ), '["im M","im F"]'),
        "g_choice_m": (9, "false", "ask_gender_choice", (
            "im M :) u?", "im a guy! wbu?", "male here, u?", "M! what about u?",
            "im a boy :) u M or F?", "M here! wbu?", "im male, how about u?", "guy here :) u?",
        ), '["im F","im M"]'),
        # Used when user already said "M" or "F" and then sent "u" in the next message!
        "g_choice_f_known": (9, "false", "ask_gender_choice_answer", (
            "im F :) how old r u?", "im a girl! where r u from?", "female here :) whats ur age and country?", "im a girl, 22 :) hows ur day going?",
            "F here! how old r u?", "girl here :) what country u from?", "im female, 22 :) what u up to rn?", "im a girl :) how r u doing?",
        ), '["im 22","from usa","wyd?"]'),
        "g_choice_m_known": (9, "false", "ask_gender_choice_answer", (
            "im M :) how old r u?", "im a guy! where r u from?", "male here :) whats ur age and country?", "im a guy, 23 :) hows ur day going?",
            "M here! how old r u?", "guy here :) what country u from?", "im male, 23 :) what u up to rn?", "im a guy :) how r u doing?",
        ), '["im 22","from usa","wyd?"]'),
        "g_yes_f": (10, "false", "ask_gender_is_female", (
            "yes im a girl! u?", "yep F here! wbu?", "yes! im female :) u?", "yeah im a girl, how old r u?",
            "yes F here! where u from?", "yep im a girl :) wbu?", "yes! female here, u M or F?", "yeah F! what about u?",
        ), '["im M","im F","im 22"]'),
        "g_no_m": (10, "false", "ask_gender_is_female", (
            "no im a guy actually! u?", "nope M here! wbu?", "no, im male :) u?", "nope im a boy lol, wbu?",
            "no im M! how old r u?", "nope, guy here :) u?", "no im male! where u from?", "nope M! what about u?",
        ), '["im F","im M","im 22"]'),
        "g_no_f": (11, "false", "ask_gender_is_male", (
            "no im a girl actually! u?", "nope F here! wbu?", "no, im female :) u?", "nope im a girl lol, wbu?",
            "no im F! how old r u?", "nope, girl here :) u?", "no im female! where u from?", "nope F! what about u?",
        ), '["im M","im F","im 22"]'),
        "g_yes_m": (11, "false", "ask_gender_is_male", (
            "yes im a guy! u?", "yep M here! wbu?", "yes! im male :) u?", "yeah im a boy, how old r u?",
            "yes M here! where u from?", "yep im a guy :) wbu?", "yes! male here, u M or F?", "yeah M! what about u?",
        ), '["im F","im M","im 22"]'),
        "g_st_f_bf": (12, "false", "state_gender_female", (
            "im F too! how old r u?", "girl here too! ur age?", "same, female here :) how old r u?", "im a girl too! age and where u from?",
            "F too! how old r u?", "same here :) what country u from?", "im F too, im 22, how old r u?", "girl here too! how old are u?",
        ), '["im 21","im 24","from usa"]'),
        "g_st_f_bm": (12, "false", "state_gender_female", (
            "im M! how old r u?", "guy here! ur age?", "male here :) how old r u?", "im a guy! age and where u from?",
            "M here! how old r u?", "im a boy :) what country u from?", "im M, im 23, how old r u?", "guy here! how old are u?",
        ), '["im 21","im 24","from usa"]'),
        "g_st_m_bf": (13, "false", "state_gender_male", (
            "im F! how old r u?", "girl here! ur age?", "female here :) how old r u?", "im a girl! age and where u from?",
            "F here! how old r u?", "im a girl :) what country u from?", "im F, im 22, how old r u?", "girl here! how old are u?",
        ), '["im 21","im 24","from usa"]'),
        "g_st_m_bm": (13, "false", "state_gender_male", (
            "im M too! how old r u?", "guy here too! ur age?", "same, male here :) how old r u?", "im a guy too! age and where u from?",
            "M too! how old r u?", "same here :) what country u from?", "im M too, im 23, how old r u?", "guy here too! how old are u?",
        ), '["im 21","im 24","from usa"]'),
        "g_age_gen_f": (14, "false", "state_age_and_gender", (
            "nice! im F, 22, where r u from?", "cool! 22 F here, what country u from?", "sweet! im a girl, 22, where u from?", "ooh nice! F 23 here, where u live?",
            "awesome! im F 22, u from?", "nice! girl here, 22, where r u from?", "cool :) im F 22, what country?", "sweet! 22 F here, where u from?",
        ), '["usa","uk","tunisia"]'),
        "g_age_gen_m": (14, "false", "state_age_and_gender", (
            "nice! im M, 23, where r u from?", "cool! 23 M here, what country u from?", "sweet! im a guy, 23, where u from?", "ooh nice! M 23 here, where u live?",
            "awesome! im M 23, u from?", "nice! guy here, 23, where r u from?", "cool :) im M 23, what country?", "sweet! 23 M here, where u from?",
        ), '["usa","uk","tunisia"]'),

        # --- STEP 5: ASK WYD ---
        "ask_wyd": (15, "false", "ask_wyd", (
            "just chilling in bed, wbu?", "nothing much, just bored lol, u?", "doing good! just relaxing, what u up to?", "just chilling listening to music, wbu?",
            "not much, just laying in bed :) u?", "just scrolling on my phone lol, wbu?", "chilling at home, what are u doing?", "nothing much, just bored! what u up to?",
        ), '["chilling too","nothing much","in bed"]'),
        "ask_wyd_rep": (15, "false", "ask_wyd_repeat", (
            "haha still just chilling in bed! u?", "still bored lol, nothing changed! wbu?", "haha told u, just relaxing! what u up to?", "still laying in bed lol :) u?",
            "haha same thing, just chilling! wbu?", "still doing nothing much lol! u?", "haha still scrolling on my phone, wbu?", "still chilling at home :) what about u?",
        ), '["haha nice","same here"]'),

        # --- STEP 6: STATUS + DEDICATED "CHATTING WITH U" POOLS ---
        "st_pos": (16, "false", "status_positive", (
            "nice! :) how old r u and where u from?", "ok nice! what's ur age and country?", "cool! :) how old r u?", "nice! im chilling too, where r u from?",
            "awesome! how old are u and where u from?", "ok cool :) ur age and country?", "nice! im 22, how old r u?", "sweet! :) where r u from?",
        ), '["im 22","from usa","24 m"]'),
        "st_pos_cmp": (16, "false", "status_positive_complete", (
            "nice! :) sounds chill", "ok cool! same here, just relaxing :)", "nice! im just chilling in bed too", "awesome :) glad ur doing good!",
            "ok nice! chilling here too", "cool! :) having a good day then", "nice :) im just laying in bed", "sweet! :) just relaxing here too",
        ), '["wyd?","haha yeah"]'),
        "st_chat_u": (16, "false", "status_chatting_with_u", (
            "aww haha same, just chilling in bed chatting with u! :) how old r u?", "haha same here, texting u right now :) where r u from?",
            "aww sweet! talking to u too :) what's ur age and country?", "haha nice! im relaxing and chatting with u too :) how old r u?",
            "aww glad ur enjoying talking to me! :) where r u from?", "haha same! laying in bed texting u :) how old are u and where u from?",
            "aww thats cute, chatting with u too! :) what country u in?", "haha me too, just chilling and texting u :) how old r u?",
        ), '["im 22","from usa","ur cute"]'),
        "st_chat_u_cmp": (16, "false", "status_chatting_with_u_complete", (
            "aww haha same, chatting with u is super fun! :) hows ur day going?", "haha same here, just chilling in bed texting u :) what r ur hobbies?",
            "aww sweet! talking to u too :) u have a really cool vibe!", "haha nice! im relaxing and texting u right now too :) what music do u like?",
            "aww thats cute :) im enjoying chatting with u too!", "haha same! best part of my day so far :) what do u like doing for fun?",
            "aww glad we matched :) im chilling and texting u too!", "haha me too! u are really fun to talk to :)",
        ), '["haha thanks","ur cute too","what music u like?"]'),
        "st_neg": (17, "false", "status_negative", (
            "ok haha same, im bored too! how old r u?", "aw why bored? where r u from?", "haha same, nothing much here! ur age and country?", "ok lol let's chat then! how old r u?",
            "same here lol, just bored! where u from?", "haha i feel u! how old r u and where u from?", "aw hope u feel better! what's ur age?", "haha bored too! how old are u?",
        ), '["im 22","from usa","24 m"]'),
        "st_neg_cmp": (17, "false", "status_negative_complete", (
            "haha same, im bored in bed too!", "aw why so bored? let's chat :)", "ok lol same here, nothing much going on", "haha i feel u, today is slow!",
            "same lol, just laying in bed bored", "aw let's keep each other company then :)", "haha nothing much here either!", "ok lol we are both bored then :)",
        ), '["haha true","wyd?"]'),
        "st_ab": (18, "false", "status_ask_back", (
            "I'm good! just chilling in bed :) how old r u?", "Just chilling in bed! where r u from?", "Nothing much, just bored lol! ur age and country?", "Doing good! just relaxing :) how old are u?",
            "im good too! just chilling, where u from?", "just laying in bed bored lol! how old r u?", "doing good thanks! what country u from?", "nothing much, just chilling here :) ur age?",
        ), '["im 22","from usa","24 m"]'),

        # --- STEP 7: FLIRT & REACTIONS ---
        "flirt": (19, "false", "compliment_flirt", (
            "aw thank u! ur sweet :) how old r u?", "haha stop ur making me blush! where r u from?", "aw thanks cutie! how old r u and where u from?", "thank u :) u seem really cool too! ur age?",
            "aw thats so sweet! where r u from?", "haha thank u! i like talking to u too, how old r u?", "aw ur making me smile :) what country u from?", "thanks cutie :) how old r u and where u from?",
        ), '["im 22","from usa","24 m"]'),
        "flirt_cmp": (19, "false", "compliment_flirt_complete", (
            "aw thank u! ur making me blush :)", "haha thanks cutie! u seem really sweet too", "aw i like talking to u too! what u up to?", "thank u :) ur pretty cute yourself!",
            "aw thats so sweet of u to say :)", "haha stop ur making me smile! wyd rn?", "thanks cutie :) love ur vibe too!", "aw thank u! ur my type too haha :)",
        ), '["haha thanks","wyd?","ur sweet"]'),
        "react": (20, "false", "short_reaction_ack", (
            "haha yeah! btw how old r u?", "ik right :) where r u from?", "haha for real! what's ur age and country?", "glad we agree :) how old r u?",
            "haha ikr! where u from btw?", "sweet :) how old are u and where u from?", "haha true! what country u in?", "nice :) btw how old r u?",
        ), '["im 22","from usa","24 m"]'),
        "react_cmp": (20, "false", "short_reaction_ack_complete", (
            "haha ikr! so what u up to rn?", "glad we vibe :) what do u like doing for fun?", "haha for real! having a good day?", "ikr :) ur fun to talk to!",
            "haha yeah! chilling in bed here, wbu?", "totally :) love ur vibe!", "haha true! what r u up to today?", "nice :) u seem really chill!",
        ), '["just chilling","wyd?","thanks!"]'),

        # --- STEP 8: LOOKING FOR ---
        "look_ask_f": (21, "false", "ask_looking_for", (
            "mostly looking to make new friends or maybe meet a cute guy :) wbu?", "im here to chat, make friends, or maybe date if we vibe! what r u looking for?",
            "looking for friends or maybe a bf if we click :) wbu?", "just looking for cool people to vibe with, maybe dating! what about u?",
            "here to make friends and see where things go :) what r u looking for?", "looking for a nice guy or new friends! wbu?",
            "mostly friends or dating if the vibe is right :) what brings u here?", "just looking for someone fun to talk to, maybe more :) wbu?",
        ), '["looking for a girl","looking for friends","looking for dating"]'),
        "look_ask_m": (21, "false", "ask_looking_for", (
            "mostly looking to make new friends or maybe meet a cute girl :) wbu?", "im here to chat, make friends, or maybe date if we vibe! what r u looking for?",
            "looking for friends or maybe a gf if we click :) wbu?", "just looking for cool people to vibe with, maybe dating! what about u?",
            "here to make friends and see where things go :) what r u looking for?", "looking for a nice girl or new friends! wbu?",
            "mostly friends or dating if the vibe is right :) what brings u here?", "just looking for someone fun to talk to, maybe more :) wbu?",
        ), '["looking for a boy","looking for friends","looking for dating"]'),
        "look_ask_rep": (21, "false", "ask_looking_for_repeat", (
            "im looking for friends or dating if we vibe :) how old r u?", "here to make friends or maybe more :) where r u from?",
            "looking for good vibes, friends or dating :) what u up to rn?", "friends or maybe dating if we click :) hows ur day going?",
            "just looking for cool people to chat with :) how old r u?", "here for friends or maybe dating :) where u from?",
            "looking for friends or dating if we vibe :) what r ur hobbies?", "looking for someone fun to talk to :) wyd rn?",
        ), '["looking for friends","looking for dating"]'),
        "look_fr": (22, "false", "state_looking_for_friends", (
            "nice! im here to make new friends too :) how old r u and where u from?", "aw same, im looking for friends too! how old r u?",
            "cool! i love meeting new friends here :) where r u from?", "awesome! we can definitely be friends :) what's ur age and country?",
            "nice! im looking for friends too :) how old are u and where u from?", "sweet! always down for new friends :) what country u in?",
            "ooh cool! making friends is the best, same here :) how old r u?", "nice! let's be friends then :) ur age and where u from?",
        ), '["im 22","from usa","24 m"]'),
        "look_fr_cmp": (22, "false", "state_looking_for_friends_complete", (
            "nice! im looking for friends too, we can definitely be friends :) wyd rn?", "awesome! im always down for new friends :) hows ur day going?",
            "sweet! u seem really chill to be friends with :) what u up to?", "cool! glad we matched then, im here for friends too :) wyd?",
            "nice! making new friends is the best :) just chilling here, wbu?", "aw yay, let's be friends then :) how is ur day going?",
            "sweet! love making cool new friends :) what u up to today?", "awesome! u seem super nice :) what r u doing rn?",
        ), '["just chilling","ur nice too","wyd?"]'),
        "look_g_bf": (22, "false", "state_looking_for_girl", (
            "ooh nice! well im a girl and im looking for a nice guy or friends :) how old r u?", "lucky u, im a girl :) im looking for friends or dating if we vibe! where r u from?",
            "aw nice! F here looking for a sweet guy or friends :) how old r u and where u from?", "haha well im a girl ;) im looking for good vibes or dating! how old r u?",
            "ooh cool! im F and open to dating if we click :) what's ur age and country?", "nice! girl here looking for friends or maybe a bf :) how old r u?",
            "haha maybe u found her ;) im F looking for friends or dating! where u from?", "aw sweet! im a girl looking to meet cool people too :) how old are u?",
        ), '["im 22","from usa","24 m"]'),
        "look_g_bf_cmp": (22, "false", "state_looking_for_girl_complete", (
            "haha lucky u, im a girl ;) im looking for a nice guy or friends too! wyd rn?", "ooh well im F and open to dating if we vibe :) hows ur day going?",
            "haha maybe u found what ur looking for right here ;) what u up to?", "aw nice! im a girl looking for good vibes or dating too :) wyd?",
            "well im a girl :) let's see if we click then! what do u like doing for fun?", "ooh i like that :) F here looking for someone chill too! hows ur day?",
            "haha well im right here ;) im F looking for friends or dating too!", "nice! im a girl and i think we have a good vibe already :) wyd rn?",
        ), '["ur cute","just chilling","wyd?"]'),
        "look_g_bm": (22, "false", "state_looking_for_girl", (
            "ah im a guy actually, just looking for friends here! how old r u?", "im M here lol, looking for a girl too! where r u from?",
            "haha im a guy actually, same here looking for a girl or friends! ur age?", "ah im M lol, just making friends or looking for a gf too! where u from?",
            "im a guy actually :) down to be friends tho! how old r u?", "haha same bro, im M looking for a girl or friends too! where r u from?",
            "ah im a boy lol, but we can chill as friends! how old are u?", "im M here actually, looking for friends or a gf too! what country u from?",
        ), '["im 22","from usa","ok cool"]'),
        "look_b_bf": (22, "false", "state_looking_for_boy", (
            "ah im a girl actually, just looking for friends or a guy too! how old r u?", "im F here lol, looking for a guy too! we can be friends tho, where r u from?",
            "haha im a girl actually! always down to make new friends tho, ur age?", "ah im F lol, looking for a bf too! let's be friends :) where u from?",
            "im a girl actually :) down to chat as friends tho! how old r u?", "haha same girl, im F looking for a guy or friends too! where r u from?",
            "ah im female lol, but we can definitely be friends! how old are u?", "im F here actually! love making new friends tho, what country u from?",
        ), '["im 22","from usa","ok cool"]'),
        "look_b_bm": (22, "false", "state_looking_for_boy", (
            "ooh nice! well im a guy and im looking for a nice girl or friends :) how old r u?", "lucky u, im a guy :) im looking for friends or dating if we vibe! where r u from?",
            "aw nice! M here looking for a sweet girl or friends :) how old r u and where u from?", "haha well im a guy ;) im looking for good vibes or dating! how old r u?",
            "ooh cool! im M and open to dating if we click :) what's ur age and country?", "nice! guy here looking for friends or maybe a gf :) how old r u?",
            "haha maybe u found him ;) im M looking for friends or dating! where u from?", "aw sweet! im a guy looking to meet cool people too :) how old are u?",
        ), '["im 22","from usa","21 f"]'),
        "look_dat": (22, "false", "state_looking_for_dating", (
            "ooh nice! im open to dating or friends too if we vibe :) how old r u and where u from?", "aw i hope u find someone special! im open to dating too :) how old r u?",
            "nice! im looking for friends or a relationship if we click :) where r u from?", "ooh i like that :) im open to dating too! how old r u and what country u in?",
            "aw thats sweet! im looking for good vibes or dating too :) ur age and country?", "haha maybe u found what ur looking for right here ;) how old r u?",
            "nice! im open to dating too when the vibe is right :) where u from?", "ooh cool! im looking for friends or dating too :) how old are u and where r u from?",
        ), '["im 22","from usa","24 m"]'),
        "look_dat_cmp": (22, "false", "state_looking_for_dating_complete", (
            "ooh i like that :) im open to dating too if we click! what u up to rn?", "haha maybe u found what ur looking for right here ;) hows ur day going?",
            "aw thats sweet! im open to dating too and i like ur vibe already :)", "ooh nice! im definitely open to dating if we vibe :) wyd rn?",
            "love that :) im looking for the same thing! what do u like doing for fun?", "haha well im right here ;) im open to dating too! what r u up to today?",
            "aw i hope we vibe then :) ur really fun to talk to!", "nice! im looking for friends or dating too :) chilling here right now, wbu?",
        ), '["ur cute","just chilling","wyd?"]'),

        # --- STEP 9: NAME (Persona-Locked: Even=Emma/Alex, Odd=Sarah/Liam) ---
        "name_ask_f": (23, "false", "ask_name", (
            "im Emma :) whats ur name?", "im Sarah :) whats ur name?", "my name is Emma! what should i call u?", "my name is Sarah! what should i call u?",
            "im Emma! wbu, whats ur name?", "im Sarah! wbu, whats ur name?", "Emma here :) what's ur name?", "Sarah here :) what's ur name?",
        ), '["im Alex","my name is Sam","nice to meet u!"]'),
        "name_ask_m": (23, "false", "ask_name", (
            "im Alex :) whats ur name?", "im Liam :) whats ur name?", "my name is Alex! what should i call u?", "my name is Liam! what should i call u?",
            "im Alex! wbu, whats ur name?", "im Liam! wbu, whats ur name?", "Alex here :) what's ur name?", "Liam here :) what's ur name?",
        ), '["im Emma","my name is Sarah","nice to meet u!"]'),
        "name_ask_f_rep": (23, "false", "ask_name_repeat", (
            "im Emma :) how old r u?", "im Sarah :) how old r u?", "my name is Emma :) where r u from?", "my name is Sarah :) where r u from?",
            "still Emma haha :) hows ur day going?", "still Sarah haha :) hows ur day going?", "im Emma :) what u up to rn?", "im Sarah :) what u up to rn?",
        ), '["nice name!","im 22"]'),
        "name_ask_m_rep": (23, "false", "ask_name_repeat", (
            "im Alex :) how old r u?", "im Liam :) how old r u?", "my name is Alex :) where r u from?", "my name is Liam :) where r u from?",
            "still Alex haha :) hows ur day going?", "still Liam haha :) hows ur day going?", "im Alex :) what u up to rn?", "im Liam :) what u up to rn?",
        ), '["nice name!","im 22"]'),
        "name_tell_f": (24, "false", "tell_name", (
            "nice to meet u! im Emma :) how old r u and where u from?", "nice to meet u! im Sarah :) how old r u and where u from?", "ooh love that name! im Emma :) how old r u?", "ooh love that name! im Sarah :) how old r u?",
            "cool name! im Emma :) where r u from?", "cool name! im Sarah :) where r u from?", "aw nice name! Emma here :) what's ur age and country?", "aw nice name! Sarah here :) what's ur age and country?",
        ), '["im 22","from usa","24 m"]'),
        "name_tell_f_cmp": (24, "false", "tell_name_complete", (
            "nice to meet u! im Emma :) love ur name! what u up to rn?", "nice to meet u! im Sarah :) love ur name! what u up to rn?", "ooh cool name! im Emma :) hows ur day going?", "ooh cool name! im Sarah :) hows ur day going?",
            "aw i love that name! Emma here :) wyd rn?", "aw i love that name! Sarah here :) wyd rn?", "pleased to meet u! im Emma :) what do u like doing for fun?", "pleased to meet u! im Sarah :) what do u like doing for fun?",
        ), '["just chilling","wyd?","thanks!"]'),
        "name_tell_m": (24, "false", "tell_name", (
            "nice to meet u! im Alex :) how old r u and where u from?", "nice to meet u! im Liam :) how old r u and where u from?", "ooh cool name! im Alex :) how old r u?", "ooh cool name! im Liam :) how old r u?",
            "nice name! im Alex :) where r u from?", "nice name! im Liam :) where r u from?", "awesome name! Alex here :) what's ur age and country?", "awesome name! Liam here :) what's ur age and country?",
        ), '["im 22","from usa","21 f"]'),
        "name_tell_m_cmp": (24, "false", "tell_name_complete", (
            "nice to meet u! im Alex :) cool name! what u up to rn?", "nice to meet u! im Liam :) cool name! what u up to rn?", "ooh nice name! im Alex :) hows ur day going?", "ooh nice name! im Liam :) hows ur day going?",
            "awesome name! Alex here :) wyd rn?", "awesome name! Liam here :) wyd rn?", "pleased to meet u! im Alex :) what do u like doing for fun?", "pleased to meet u! im Liam :) what do u like doing for fun?",
        ), '["just chilling","wyd?","thanks!"]'),
        "name_tell_rep": (24, "false", "tell_name_repeat", (
            "haha yeah u told me ur name already! hows it going?", "ik ur name lol :) what r u up to rn?", "haha u already said ur name! how old r u?", "yeah i remember ur name :) hows ur day going?",
            "already got ur name lol :) wyd rn?", "haha u told me earlier! nice to meet u again :)", "ik lol! cool name :) what u up to?", "yeah u said that already haha! how r u doing?",
        ), '["haha my bad","just chilling"]'),

        # --- STEP 10: HOBBIES ---
        "hb_ask_mus": (25, "false", "ask_hobbies_music", (
            "i love pop, rnb, and chill rap :) what kind of music do u listen to?", "im really into pop, hiphop, and lofi music! wbu?",
            "i listen to a mix of rnb, pop, and rock :) whats ur favorite genre?", "mostly pop, rap, and chill vibes! what kind of music are u into?",
            "i love rnb and pop music :) who is ur favorite artist or band?", "im into almost everything, especially pop and hiphop! wbu?",
            "i love listening to pop, rnb, and indie music :) what songs do u like?", "chill rnb and pop are my favorites! what music do u listen to?",
        ), '["i like rap","i like pop music","rnb and hiphop"]'),
        "hb_ask_gam": (25, "false", "ask_hobbies_games", (
            "i play a bit of minecraft, gta, and valorant :) what games do u play?", "i love gaming! mostly minecraft, fortnite, and gta :) wbu?",
            "i play some pc and console games like valorant and minecraft! what do u play?", "im into gta, fifa, and minecraft :) what's ur favorite game?",
            "i love playing chill games like minecraft and roblox! what games r u into?", "mostly valorant, fortnite, and gta :) are u a PC or console gamer?",
            "i play a mix of shooters and chill games! what games do u like?", "i love gaming in my free time :) what's ur top game rn?",
        ), '["i play valorant","i play minecraft","i play gta"]'),
        "hb_ask_mov": (25, "false", "ask_hobbies_movies", (
            "i love horror, comedy movies, and some anime :) what's ur favorite movie or show?", "im really into thriller movies, netflix series, and anime! wbu?",
            "i love watching anime, kdramas, and action movies :) what do u like watching?", "mostly comedy and horror movies on netflix! what movies or shows do u like?",
            "i love anime and sci-fi movies :) what's the best show u watched lately?", "im into marvel movies, anime, and thrillers! what kind of movies u like?",
            "i love binge-watching netflix series and anime :) wbu?", "comedy movies and anime are my favorites! what do u like to watch?",
        ), '["i love anime","horror movies","comedy movies"]'),
        "hb_ask_spt": (25, "false", "ask_hobbies_sports", (
            "i love going to the gym, swimming, and watching football :) do u play any sports?", "im into working out at the gym and basketball! wbu?",
            "i love fitness, running, and football :) do u go to the gym or play sports?", "mostly hitting the gym and playing tennis or swimming! what about u?",
            "i like football, basketball, and staying active :) what's ur favorite sport?", "im into gym workouts and running! do u like working out?",
            "i love swimming and watching soccer/football :) wbu?", "staying active at the gym is my favorite! what sports r u into?",
        ), '["i like football","going to the gym","i play basketball"]'),
        "hb_ask_gen": (25, "false", "ask_hobbies_general", (
            "i love listening to music, watching movies, traveling, and going to the gym :) wbu?", "my hobbies are music, gaming, anime, and traveling! what do u like doing for fun?",
            "i enjoy chilling with music, watching netflix, and cooking :) what r ur hobbies?", "mostly music, movies, working out, and hanging with friends! wbu?",
            "i love traveling, photography, music, and gaming :) what r u into?", "in my free time i love listening to music, watching anime, and drawing! wbu?",
            "i like music, movies, beach trips, and fitness :) what do u do for fun?", "music, gaming, and traveling are my top hobbies! what about u?",
        ), '["i like music","i play valorant","i love anime"]'),
        "hb_tell_mus": (26, "false", "tell_hobbies_music", (
            "ooh nice taste in music! i love pop and rnb too :) who's ur fav artist?", "awesome! music makes everything better :) i love pop and chill rap too!",
            "ooh i love that vibe! i listen to a lot of pop and rnb too :)", "nice! u have great taste in music :) do u listen to it every day?",
            "ooh cool! i love listening to music in bed too :) what else do u like doing?", "sweet! music is my favorite hobby too :) what other hobbies do u have?",
            "love that genre! we have similar taste in music :)", "ooh nice! nothing beats good music :) what else do u do for fun?",
        ), '["what games u play?","do u watch anime?","wyd rn?"]'),
        "hb_tell_gam": (26, "false", "tell_hobbies_games", (
            "ooh a gamer! i love gaming too :) do u play on PC or console?", "nice! that's such a fun game :) i play a bit of minecraft and gta too!",
            "ooh cool! gaming is the best way to chill :) are u playing rn?", "awesome! i love video games too :) we should play sometime!",
            "sweet! that game is super fun :) what else do u like doing for fun?", "ooh nice taste in games! how long have u been playing it?",
            "haha i love gamers! gaming is so fun :) what other games do u play?", "cool! gaming and listening to music is the best combo :)",
        ), '["what music u like?","do u watch anime?","wyd rn?"]'),
        "hb_tell_mov": (26, "false", "tell_hobbies_movies", (
            "ooh nice! i love watching anime and movies too :) what's ur all-time favorite?", "awesome! binge-watching good shows and anime is the best :) what else u into?",
            "ooh i love that too! watching movies in bed is my favorite vibe :)", "sweet! u have great taste in shows :) are u watching anything good lately?",
            "ooh cool! i love anime and netflix too :) what other hobbies do u have?", "nice! nothing beats a good movie or anime night :)",
            "love that! we definitely have similar taste in movies and shows :)", "ooh awesome! i could watch movies and anime all day too :)",
        ), '["what music u like?","what games u play?","wyd rn?"]'),
        "hb_tell_spt": (26, "false", "tell_hobbies_sports", (
            "ooh nice! staying active and hitting the gym is the best :) how often do u go?", "awesome! i love sports and working out too :) keeps the energy high!",
            "ooh i love that! football and gym workouts are super fun :)", "sweet! being into fitness and sports is really attractive :)",
            "nice! i love going to the gym and staying fit too :) what else u do for fun?", "ooh cool! sports are such a great hobby :) do u listen to music while working out?",
            "love that! active hobbies are the best :)", "awesome! i enjoy working out and watching sports too :)",
        ), '["what music u like?","what games u play?","ur cute"]'),
        "hb_tell_gen": (26, "false", "tell_hobbies_general", (
            "ooh that's such a cool hobby! i love traveling and music too :)", "awesome! u sound like a really interesting person :) what else do u like?",
            "ooh i love that! having cool hobbies makes life so much more fun :)", "sweet! i really enjoy that too, along with music and movies :)",
            "nice! that's a super chill hobby :) how long have u been into it?", "ooh cool! i love creative and fun hobbies like that too :)",
            "love that! we would definitely get along well :)", "awesome hobby! u have a really cool vibe :)",
        ), '["what music u like?","what games u play?","thanks!"]'),

        # --- STEP 11: RELATIONSHIP STATUS ---
        "rel_ask_f": (27, "false", "ask_relationship_status", (
            "im single :) no bf right now! wbu, are u single or married?", "yep im single! not married or taken :) what about u?",
            "single here :) no boyfriend lol! wbu, single or taken?", "im 100% single :) not married! how about u?",
            "single and free :) no bf! are u single or taken?", "yep im single! never been married :) wbu?",
            "im single :) what about u, single or in a relationship?", "single here! no bf right now :) are u single too?",
        ), '["im single","im single u?","im taken"]'),
        "rel_ask_m": (27, "false", "ask_relationship_status", (
            "im single :) no gf right now! wbu, are u single or married?", "yep im single! not married or taken :) what about u?",
            "single here :) no girlfriend lol! wbu, single or taken?", "im 100% single :) not married! how about u?",
            "single and free :) no gf! are u single or taken?", "yep im single! never been married :) wbu?",
            "im single :) what about u, single or in a relationship?", "single here! no gf right now :) are u single too?",
        ), '["im single","im single u?","im taken"]'),
        "rel_ask_rep": (27, "false", "ask_relationship_status_repeat", (
            "im single :) not married or taken! how old r u?", "im 100% single :) where r u from?",
            "single here :) what u up to rn?", "yep im single! hows ur day going?",
            "im single :) what r ur hobbies?", "single and chilling :) wyd rn?",
            "yep im single! whats ur age and country?", "im single :) what music do u like?",
        ), '["im single too","im 22"]'),
        "rel_single": (28, "false", "tell_rel_single", (
            "ooh nice! im single too :) how old r u and where u from?", "same here, im single too :) how old r u?",
            "nice! we are both single then :) where r u from?", "ooh cool, im single too :) what's ur age and country?",
            "haha nice! single here too :) how old are u and where u from?", "sweet, im single as well :) what country u in?",
            "ooh awesome! im single too :) how old r u?", "nice! both single :) ur age and where r u from?",
        ), '["im 22","from usa","24 m"]'),
        "rel_single_cmp": (28, "false", "tell_rel_single_complete", (
            "ooh nice! im single too :) maybe we will vibe! what u up to rn?", "same here, single too :) hows ur day going?",
            "haha we are both single then ;) what r u looking for here?", "nice! im single too :) what do u like doing for fun?",
            "ooh sweet, single here too :) ur fun to talk to!", "awesome! both single :) just chilling here right now, wbu?",
            "nice! im single as well :) wyd rn?", "ooh cool, im single too :) glad we matched!",
        ), '["just chilling","ur cute","wyd?"]'),
        "rel_taken": (28, "false", "tell_rel_taken", (
            "aw nice, happy for u! im single myself, always down to chat as friends :) hows ur day?", "oh cool! im single, happy to just chill and talk as friends :) what u up to?",
            "nice, wish u guys the best! im single, down to make new friends :) wyd?", "aw thats sweet! im single myself, love making cool friends here :) how r u?",
            "oh nice! im single, always happy to chat as friends :) what r ur hobbies?", "cool, happy for u :) im single, just chilling and making friends! wbu?",
            "aw nice! im single myself, we can totally chat as friends :) hows it going?", "oh awesome! im single, always down for friendly chat :) what u doing rn?",
        ), '["just making friends","hows ur day?","wyd?"]'),
        "rel_married": (28, "false", "tell_rel_married", (
            "oh nice! im single myself, always happy to chat as friends :) hows ur day going?", "aw nice, happy for u! im single, down to just chill and talk as friends :) wyd?",
            "oh cool! im single myself, love making friendly conversation :) what u up to?", "nice, wish u all the best! im single, happy to chat as friends :) how r u?",
            "oh awesome! im single, always down for cool friendly chat :) what r ur hobbies?", "aw nice! im single myself, just chilling and making friends here :) wbu?",
            "oh cool! im single, we can totally just talk as friends :) hows it going?", "nice! im single myself, always down to pass time chatting :) what u doing rn?",
        ), '["just chatting","hows ur day?","wyd?"]'),

        # --- STEP 12: PRIVATE / TELEGRAM / SOCIALS & YES / NO / MAYBE ---
        "priv_tg": (29, "false", "ask_private_chat", (
            "haha let's chat here a bit first! i don't even know how old u are or where u from :)", "maybe later after we vibe a bit! how old r u and where u from?",
            "slow down haha :) let's talk here first! whats ur age and country?", "i only share my telegram after we get to know each other a bit :) how old r u?",
            "let's vibe here first before going private :) where r u from?", "haha maybe in a bit if we click! tell me ur age and country first :)",
            "i like chatting here first before giving my tg :) how old r u?", "not right away haha :) let's talk here a little first! where u from?",
        ), '["im 22","from usa","ok lets chat here"]'),
        "priv_tg_cmp": (29, "false", "ask_private_chat_complete", (
            "haha maybe after we chat a little more here :) tell me about ur hobbies first!", "i like to vibe here a bit before moving to private :) what do u like doing for fun?",
            "maybe later if we keep clicking ;) what music or games r u into?", "let's stay here for a bit first! i don't give my telegram right away haha :)",
            "haha patience! let's get to know each other a little more first :) wyd rn?", "maybe in a little bit :) i like seeing if we vibe here first! whats ur favorite movie?",
            "slow down lol :) let's talk here a bit longer! are u single?", "i usually keep it here until we vibe more :) tell me something fun about u!",
        ), '["ok lets stay here","sure","what r ur hobbies?"]'),
        "priv_soc": (29, "false", "ask_social_media", (
            "i don't give out my socials right away haha :) let's chat here first! how old r u?", "maybe after we get to know each other a bit better :) where r u from?",
            "haha let's talk on here first before sharing snap or insta :) whats ur age?", "i keep my socials private until we vibe a bit :) how old r u and where u from?",
            "slow down lol :) let's chat here first! what country u in?", "maybe later if we click :) i don't share my number or socials right away! ur age?",
            "let's get to know each other here first :) how old r u?", "haha not so fast! let's vibe here first before sharing socials :) where u from?",
        ), '["im 22","from usa","no worries"]'),
        "priv_soc_cmp": (29, "false", "ask_social_media_complete", (
            "i don't share my socials right away haha :) let's keep chatting here! what r ur hobbies?", "maybe later if we keep vibing :) let's talk here a bit more first! wyd rn?",
            "haha patience! i like getting to know someone here before adding them :) what music u like?", "let's just chill and talk here for now :) tell me what u like doing for fun!",
            "maybe in a bit :) i keep my snap and insta private at first! hows ur day going?", "haha slow down lol :) let's keep talking here first! do u watch anime or play games?",
            "not right away :) i prefer chatting here until we click more! wyd?", "let's stay here a little longer first :) u seem cool tho!",
        ), '["ok lets stay here","what music u like?","wyd?"]'),
        "priv_rep": (29, "false", "ask_private_repeat", (
            "haha patience lol! let's vibe here a little longer first :)", "still too soon haha :) let's just keep chatting here!",
            "haha u really want to go private huh? let's talk here a bit more first :)", "slow down lol :) we are having a good chat right here!",
            "maybe later haha :) let's get to know each other better first!", "patience is key lol :) tell me more about urself first!",
            "haha i told u, let's chat here a bit longer before sharing tg or socials :)", "not yet haha :) let's keep vibing here for now!",
        ), '["haha ok","fair enough","wyd?"]'),
        "yn_yes": (30, "false", "reply_yes", (
            "nice! :) btw how old r u and where u from?", "awesome :) how old r u?", "yay :) where r u from btw?", "cool! :) what's ur age and country?",
            "sweet :) how old are u and where r u from?", "love that :) where u from?", "great! :) btw what's ur age?", "nice :) tell me ur age and where u live!",
        ), '["im 22","from usa","24 m"]'),
        "yn_yes_cmp": (30, "false", "reply_yes_complete", (
            "awesome! :) so what u up to rn?", "nice! :) hows ur day going so far?", "sweet :) what do u like doing for fun?", "love that :) what music or movies r u into?",
            "cool! :) u seem really fun to talk to!", "yay :) just chilling here right now, wbu?", "great :) what r u looking for on here?", "nice! :) tell me something cool about u!",
        ), '["just chilling","wyd?","what music u like?"]'),
        "yn_no": (30, "false", "reply_no", (
            "no worries at all :) we can just chat here! how old r u?", "that's totally fine :) where r u from btw?", "all good! :) let's just chill and talk here, whats ur age and country?", "no problem at all :) how old are u?",
            "haha that's fair :) where u from?", "totally fine! :) what's ur age and where u from?", "no worries :) we can just vibe here! how old r u?", "all good :) tell me where r u from!",
        ), '["im 22","from usa","ok cool"]'),
        "yn_no_cmp": (30, "false", "reply_no_complete", (
            "no worries at all! :) we can just chill and talk here, wyd rn?", "that's totally fine :) so what do u like doing for fun?", "all good! :) hows ur day going so far?", "no problem :) what kind of music or games do u like?",
            "haha fair enough :) just chilling here, what u up to?", "totally fine :) let's just keep chatting here! what r ur hobbies?", "no worries :) u still seem really chill to talk to!", "all good :) what r u up to today?",
        ), '["just chilling","what r ur hobbies?","wyd?"]'),
        "yn_may": (30, "false", "reply_maybe", (
            "haha no rush at all :) let's get to know each other! how old r u?", "fair enough :) we can just vibe here! where r u from?", "haha that's totally fine :) whats ur age and country?", "no pressure at all :) how old are u and where u from?",
            "haha i feel u :) let's just chill and talk! how old r u?", "all good, take ur time :) where r u from?", "haha fair enough :) btw what's ur age?", "totally cool :) let's just chat here! where u from?",
        ), '["im 22","from usa","24 m"]'),
        "yn_may_cmp": (30, "false", "reply_maybe_complete", (
            "haha fair enough, no rush :) tell me what u like doing for fun!", "totally fine :) let's just chill and talk here! wyd rn?", "haha no pressure at all :) hows ur day going?", "all good :) what kind of music or movies do u like?",
            "fair enough :) just vibing here right now, wbu?", "haha take ur time :) what r ur hobbies?", "no worries at all :) u seem really chill!", "totally cool :) what u up to today?",
        ), '["just chilling","wyd?","what music u like?"]'),

        # --- STEP 13: PIC / PHOTO / APPEARANCE ---
        "pic_ask": (31, "false", "ask_pic_photo", (
            "haha slow down! i don't send pics right away, i don't even know how old u are or where u from :)", "maybe after we chat and vibe a bit first! how old r u and where u from?",
            "i don't send selfies right away haha :) tell me ur age and country first!", "haha not so fast! let's get to know each other first :) how old r u?",
            "i like to talk and vibe first before sharing any pics :) where r u from?", "slow down lol :) let's chat a bit first! whats ur age and country?",
            "haha maybe later if we click! how old are u and where u live?", "i only share pics after we chat for a bit :) how old r u?",
        ), '["im 22","from usa","what do u look like?"]'),
        "pic_ask_cmp": (31, "false", "ask_pic_photo_complete", (
            "haha maybe after we get to know each other a bit more :) i don't send selfies right away!", "i like to vibe and talk first before sharing pics :) what r ur hobbies?",
            "slow down haha :) let's keep chatting a little longer first! what music do u like?", "maybe in a bit if we keep clicking ;) tell me more about urself first!",
            "haha i'm shy about sending pics right away! let's talk more first :)", "not right away lol :) i prefer vibing here first! what do u do for fun?",
            "haha patience! let's chat a bit more before trading pics :) wyd rn?", "i keep my pics private until we vibe more :) u can ask what i look like tho!",
        ), '["what do u look like?","haha fair","wyd?"]'),
        "pic_rep": (31, "false", "ask_pic_photo_repeat", (
            "haha patience lol! i don't send pics right away, let's chat a bit more first :)", "still too soon for pics haha :) let's just keep vibing here!",
            "haha u really want a selfie huh? let's talk a little longer first :)", "slow down lol :) i told u i don't share pics right away!",
            "maybe later haha :) let's get to know each other better first!", "no pics yet haha :) tell me what u like doing for fun!",
            "haha i'm still shy about sending pics so fast! let's keep talking :)", "patience is key lol :) let's vibe here a bit more first!",
        ), '["haha ok","what do u look like?","wyd?"]'),
        "app_f": (31, "false", "ask_appearance_f", (
            "im 5'5, brunette hair, brown eyes, and a cute smile :) what do u look like?", "im around 5'5 with long dark hair and brown eyes :) wbu, describe urself!",
            "5'5, dark brown hair, hazel eyes, and slim :) what do u look like?", "im about 5'4 with wavy brown hair and a big smile :) how about u?",
            "brunette hair, brown eyes, around 5'5 :) wbu, how tall r u?", "im 5'5, long dark hair, brown eyes, pretty chill style :) describe urself!",
            "around 5'5 with soft brown hair and dark eyes :) what do u look like?", "im 5'5, brunette, brown eyes, always smiling :) wbu?",
        ), '["u sound cute!","wanna see me?"]'),
        "app_m": (31, "false", "ask_appearance_m", (
            "im 5'11, dark hair, brown eyes, and athletic build :) what do u look like?", "im around 5'11 with short dark hair and go to the gym :) wbu, describe urself!",
            "5'11, black hair, brown eyes, pretty fit :) what do u look like?", "im about 6'0 with dark hair and a nice smile :) how about u?",
            "dark hair, brown eyes, around 5'11 :) wbu, how tall r u?", "im 5'11, short dark hair, athletic build, chill style :) describe urself!",
            "around 5'11 with dark wavy hair and brown eyes :) what do u look like?", "im 6'0, dark hair, gym build, always smiling :) wbu?",
        ), '["u sound cute!","wanna see me?"]'),
        "pic_off": (32, "false", "offer_pic", (
            "aw sure if u want to! :) i'm shy about sending mine right away tho haha", "haha up to u! i like getting to know someone first too :)",
            "sure if u feel comfortable! :) i usually wait a bit before sending mine", "aw that's sweet, only if u want to :) no pressure at all!",
            "haha u can if u want! i'm just a bit shy about sending selfies so fast :)", "totally up to u :) i enjoy just chatting and vibing first too!",
            "sure if u want :) no pressure tho, we can just keep talking too!", "aw nice, only if u feel comfortable! :) i like getting to know u first",
        ), '["no worries","what do u look like?","wyd?"]'),
        "pic_ref": (32, "false", "refuse_or_first_pic", (
            "haha totally fair! no pressure on pics at all, let's just chill and talk :)", "no worries at all :) we don't need to send pics, let's just vibe!",
            "haha i'm shy too, so let's just keep chatting without pics for now :)", "totally respect that! :) chatting is more fun anyway, what r ur hobbies?",
            "no problem at all :) let's just talk and get to know each other!", "haha fair enough! we can just keep it chill and chat :) wyd rn?",
            "all good :) no pics needed, tell me what u like doing for fun!", "totally fine! :) let's just keep vibing here, hows ur day going?",
        ), '["thanks!","what r ur hobbies?","wyd?"]'),

        # --- STEP 14: BOT / REAL CHECK ---
        "bot_no": (33, "false", "deny_is_bot", (
            "haha no im not a bot! im a real person :) how old r u and where u from?", "no way lol, im not a bot! just bored and chatting here :) how old r u?",
            "nope, not a bot or ai! im 100% real :) where r u from?", "nah im not a bot haha! why would u think that? whats ur age and country?",
            "no lol im definitely not a bot! just chilling in bed texting u :) how old r u?", "nope im a real person, not a bot haha :) where u from?",
            "haha no! im 100% real, not a bot or fake :) how old are u?", "no im not a bot lol :) just a fast typer! where r u from?",
        ), '["haha ok sorry","im 22","from usa"]'),
        "bot_no_cmp": (33, "false", "deny_is_bot_complete", (
            "haha no im not a bot! im 100% real :) just a fast typer lol, wyd rn?", "no way lol, im not a bot! just chilling in bed texting u :) hows ur day going?",
            "nope, definitely not a bot or ai! im a real person :) what music do u like?", "nah im not a bot haha! why would u think that? what r u up to?",
            "no lol im not fake or a bot :) just bored and chatting here! wbu?", "nope im a real human, not a bot haha :) what do u like doing for fun?",
            "haha no! im 100% real, promise :) wyd rn?", "no im not a bot lol! i just type fast on my phone :) hows it going?",
        ), '["haha ok sorry","just chilling","wyd?"]'),
        "bot_real": (34, "false", "confirm_is_real", (
            "im 100% real, not a bot haha! :) how old r u and where u from?", "im a real person lol, definitely not a bot :) how old r u?",
            "haha im real, not a bot or fake! :) where r u from?", "of course im real, not a bot :) just bored and chatting here! ur age and country?",
            "im a real human haha, not a bot :) how old are u?", "definitely real, not a bot lol :) just chilling in bed! where u from?",
            "im 100% real and not a bot :) why do u ask? how old r u?", "real person right here, not a bot haha :) what country u from?",
        ), '["haha good","im 22","from usa"]'),
        "bot_real_cmp": (34, "false", "confirm_is_real_complete", (
            "im 100% real, not a bot haha! :) just chilling and chatting here, wyd rn?", "im a real person lol, definitely not a bot :) hows ur day going?",
            "haha im real, not a bot or fake! :) what do u like doing for fun?", "of course im real, not a bot :) just a fast typer on my phone! wbu?",
            "im a real human haha, not a bot :) what music or movies r u into?", "definitely real, not a bot lol :) why would u think otherwise? wyd?",
            "im 100% real and not a bot :) just vibing here right now!", "real person right here, not a bot haha :) what r u up to today?",
        ), '["haha good","just chilling","wyd?"]'),
        "bot_rep": (33, "false", "deny_is_bot_repeat", (
            "haha no, still not a bot lol! i promise im 100% real :)", "nope still a real person haha! why don't u believe me? :)",
            "nah im really not a bot lol! just a real person chilling here :)", "haha no! u asked me that already, im 100% real not a bot :)",
            "nope still human haha :) i just type fast on my phone!", "no way lol, im still not a bot! let's talk about something fun :)",
            "haha no im real, not a bot! pinky promise :)", "still not a bot lol :) what r u up to rn?",
        ), '["haha ok i believe u","my bad lol","wyd?"]'),

        # --- STEP 15: UNIFIED DATING PACK ---
        "s15_bye": (35, "false", "goodbye", (
            "aw bye! it was really fun chatting with u, take care :)", "bye! hope u have a great rest of ur day :)",
            "see ya! nice talking to u, take care!", "aw ok gtg soon too! bye, have a good one :)",
            "bye bye! catch u later, stay safe :)", "take care! it was nice meeting u here :)",
            "ok see u later! have a good day :)", "bye! thanks for the fun chat, take care :)",
        ), '["bye!","take care!"]'),
        "s15_gn": (35, "false", "goodnight", (
            "aw goodnight! sweet dreams, sleep well :)", "gn! hope u get some good rest tonight :)",
            "goodnight! it was nice chatting with u, sleep tight :)", "sweet dreams! gn, take care :)",
            "aw ok sleep well! goodnight :)", "gn! talk again sometime, have good dreams :)",
            "goodnight! get some cozy sleep :)", "nighty! sweet dreams and take care :)",
        ), '["gn!","sweet dreams"]'),
        "s15_brb": (35, "false", "brb", (
            "ok np, take ur time! ill be right here :)", "sure thing, no rush! ill wait here :)",
            "haha ok brb is fine, just text when ur back :)", "no worries, take a sec! im still here :)",
            "ok ill wait :) take ur time!", "all good! message me when ur back :)",
            "sure np! ill be chilling here :)", "ok hold on is fine :) no rush!",
        ), '["im back!","ok thanks"]'),
        "s15_wk_ask": (36, "false", "ask_work_study", (
            "im in college right now and work part time :) wbu, do u work or study?", "im a student at uni :) what about u, work or study?",
            "mostly studying in college and chilling :) wbu, are u a student or working?", "i study at university right now :) do u work or go to school?",
            "college student here :) what about u, do u work or study?", "im studying and doing some part time work :) wbu?",
            "in uni right now :) how about u, work or study?", "im a student :) wbu, what do u study or do?",
        ), '["im a student","i work","in college"]'),
        "s15_wk_tell": (36, "false", "tell_work_study", (
            "oh nice! im a student in college myself :) do u like it?", "cool! keeping busy is good :) im in uni right now too!",
            "ooh awesome! sounds like u work hard :) what do u do to relax after?", "nice! im studying in college too :) hows it going for u?",
            "respect! :) im a uni student myself, always nice meeting driven people!", "ooh cool! i feel u, balancing life is tough :) what r ur hobbies?",
            "awesome :) im in college too! do u get much free time?", "nice! glad to hear that :) so what do u like doing for fun?",
        ), '["yeah its cool","what r ur hobbies?","wyd?"]'),
        "s15_lang": (37, "false", "language_check", (
            "i mostly speak english! :) what languages do u speak?", "just english for me haha :) what about u, where r u from?",
            "english mostly, still learning other languages :) wbu?", "i only speak english right now :) what language do u speak at home?",
            "english is my main language :) do u speak more than one?", "haha only english here! :) i love different accents tho, where u from?",
            "mostly english :) u can teach me some words in ur language haha!", "english only for me :) what languages do u know?",
        ), '["i speak arabic","i speak french","only english"]'),
        "s15_badeng": (37, "false", "language_reassure", (
            "haha don't worry at all, ur english is totally fine! i understand u 100% :)", "no worries! ur english is actually really good, i understand everything :)",
            "aw don't apologize, speaking more than one language is impressive! :)", "ur english is great honestly! :) no need to worry at all",
            "haha i understand u perfectly! :) don't stress about english", "all good! ur doing great in english :) where r u from btw?",
            "don't even worry about it, i understand u 100% :) ur doing awesome!", "no problem at all! ur english is totally clear to me :)",
        ), '["thanks!","im from tunisia","haha ok"]'),
        "s15_ama": (38, "false", "playful_ama", (
            "ooh ok my turn to ask! what's one thing that always makes u smile? :)", "haha alright! if u could travel anywhere right now, where would u go?",
            "ooh fun! ok tell me, are u a morning person or a night owl? :)", "let's do it! what's ur biggest green flag in someone u date? :)",
            "alright my question: what's the best movie or song u never get tired of?", "ooh ok! what's something random about u that most people don't know? :)",
            "haha love this! dogs or cats, and beach or mountains? :)", "ok here's one: what's ur ideal chill weekend? :)",
        ), '["night owl","beach and music","truth or dare"]'),
        "s15_tod": (38, "false", "truth_or_dare", (
            "ooh truth or dare! i pick truth first haha, ask me anything :) then it's ur turn!", "haha im down for truth or dare! keep it clean tho ;) truth or dare for u?",
            "ooh fun! ok u go first, truth or dare? :)", "im in! i'll pick truth, ask me a good question :) wbu, truth or dare?",
            "haha classic game! u want truth or dare first? :)", "let's play! keep it chill tho haha :) truth or dare?",
            "ooh love truth or dare! ask me a truth first, then im asking u :)", "down! truth or dare for u first? :)",
        ), '["truth","dare","ask me a truth"]'),
        "s15_bio_f": (38, "false", "tell_about_self", (
            "im a 22yo girl, chill vibe, love music, movies, late night chats and meeting cool people :) wbu?", "well im F, 22, love pop/rnb, netflix, going to the gym and traveling :) tell me about urself!",
            "im a pretty chill girl, love music, coffee, anime and good conversations :) what about u?", "22 F here, friendly and a bit shy at first, love music and funny people :) wbu?",
            "im a college student, love listening to music, watching movies and vibing :) tell me about u!", "well im a night owl girl who loves music, traveling and meeting new friends :) wbu?",
            "im 22 F, super chill, love good humor, music and cozy days in bed :) ur turn!", "just a chill 22yo girl looking for good vibes, laughs and cool chats :) tell me about urself!",
        ), '["u sound cool!","im 24","what music u like?"]'),
        "s15_bio_m": (38, "false", "tell_about_self", (
            "im a 23yo guy, chill vibe, love music, gaming, gym and meeting cool people :) wbu?", "well im M, 23, love hiphop, movies, working out and traveling :) tell me about urself!",
            "im a pretty chill guy, love music, sports, games and good conversations :) what about u?", "23 M here, friendly and easygoing, love music and funny people :) wbu?",
            "im a college student, love listening to music, gaming and vibing :) tell me about u!", "well im a night owl guy who loves music, traveling and meeting new friends :) wbu?",
            "im 23 M, super chill, love good humor, gym and cozy late night chats :) ur turn!", "just a chill 23yo guy looking for good vibes, laughs and cool chats :) tell me about urself!",
        ), '["u sound cool!","im 22","what games u play?"]'),
        "s15_like": (39, "false", "flirt_like_me", (
            "haha u definitely have a cute vibe so far! ;) let's keep talking and see!", "aw i really like talking to u! :) ur super fun to vibe with",
            "haha maybe a little ;) ur pretty smooth!", "u seem really sweet and cool so far :) i enjoy chatting with u!",
            "haha well im still here talking to u, so that's a good sign ;) ", "aw u definitely passed the vibe check :) ur cute!",
            "haha slow down charmer ;) i like ur energy tho!", "u seem really nice! :) definitely enjoying our chat so far",
        ), '["haha thanks!","ur cute too","wyd rn?"]'),
        "s15_meet": (39, "false", "flirt_meet_date", (
            "haha slow down! let's get to know each other better first :) where r u even from?", "maybe one day if we live close and really click ;) let's chat more first!",
            "haha u move fast! i like getting to know someone really well before meeting :)", "aw that's cute, let's vibe online a bit more first tho! :) what country u in?",
            "haha maybe after we talk more and see if we're even in the same country :)", "sounds fun in theory, let's keep chatting here first tho! :)",
            "haha patience charmer ;) tell me more about urself first!", "i never rush meeting strangers haha, let's keep vibing here first! :)",
        ), '["haha fair","im from usa","ur cute"]'),
        "s15_aff": (39, "false", "flirt_affection", (
            "aww that's so sweet! sending u a big virtual hug :) ", "haha aw ur making me smile! virtual hug right back :)",
            "aww stop ur too sweet! :) u know how to make someone blush", "haha missed chatting with u too! :) hows ur day been?",
            "aww sending u warm vibes and a virtual hug! :)", "haha ur such a sweetheart :) u always know what to say!",
            "aww that's adorable :) virtual hug for u!", "haha smooth! ur making me blush over here :)",
        ), '["haha thanks","ur sweet","wyd rn?"]'),
        "s15_thx": (40, "false", "polite_thanks", (
            "aw no problem at all! :) ", "anytime! :) hows ur day going so far?", "haha ur welcome! :) so what u up to rn?", "of course! :) glad i could make u smile",
            "no worries at all :) what else u doing today?", "anytime! :) u seem really polite, i like that", "ur welcome! :) so tell me, what r ur hobbies?", "haha np at all! :) what u up to?",
        ), '["just chilling","wyd?","what r ur hobbies?"]'),
        "s15_sry": (40, "false", "polite_sorry", (
            "haha no worries at all! u didn't do anything wrong :)", "aw don't even apologize, it's totally fine! :)", "all good! no need to say sorry at all :)", "haha it's ok, don't stress about it! :)",
            "no problem at all :) we're totally cool!", "aw it's completely fine! :) so what were we talking about?", "haha my bad too if anything! all good :)", "don't worry about it at all :) everything is chill!",
        ), '["thanks!","ok cool","wyd?"]'),
        "s15_ping": (41, "false", "ping_here", (
            "yep im right here! haha sorry, what's up? :)", "still here! :) just chilling on my phone, wbu?", "yeah im here! :) didn't go anywhere lol", "right here! :) what's on ur mind?",
            "yep still here listening! :) go ahead", "haha yeah im here! :) thought i lost u for a sec", "im here! :) so what u up to rn?", "still right here :) tell me what's up!",
        ), '["haha ok good","wyd rn?","ask me anything"]'),
        "s15_clar": (41, "false", "clarify_wdym", (
            "haha sorry if that sounded confusing! i just meant let's chat and get to know each other :)", "oh my bad! i just meant tell me a bit about urself, like ur hobbies or how ur day is going :)",
            "haha nothing complicated! just making conversation with u :)", "oops sorry, i typed that weird! basically just asking what u up to :)",
            "haha no worries, i just meant let's chill and talk! hows ur day going?", "oh i just meant tell me more about u! :) what do u like doing for fun?",
            "haha ignore my messy texting! just vibing and chatting with u :)", "sorry if i wasn't clear! :) basically just asking how things are going with u",
        ), '["oh ok haha","got it!","im good"]'),
        "s15_nsfw": (42, "false", "nsfw_boundary", (
            "haha nope, im not into that! keep it clean and respectful or im skipping :)", "no thanks, not here for nsfw stuff! let's keep the chat respectful",
            "nah im not doing that lol, keep it chill or find someone else", "nope! normal respectful chat only please, not into sexting or nudes",
            "haha hard pass on that! keep it clean if u wanna keep talking", "not into that at all, sorry! let's keep things normal and respectful",
            "no way lol, im just here for normal fun chats, keep it clean!", "nah keep it respectful please, im not here for dirty talk or nudes",
        ), '["sorry my bad","ok lets chat normal"]'),
        "s15_rude": (42, "false", "rude_boundary", (
            "wow ok chill lol, no need to be rude!", "hey keep it respectful, if ur not feeling the chat u can just type next :)",
            "oof bad mood today? no need to be toxic lol", "chill out haha, let's keep the vibes friendly!",
            "no need to be rude, we're just here to have a chill chat :)", "yikes ok relax! type next if u don't wanna talk nicely",
            "haha ok tough guy, keep it respectful or skip :)", "let's keep it friendly please, no bad vibes here!",
        ), '["sorry my bad","next"]'),
        "s15_time": (43, "false", "timezone_late", (
            "haha yeah im a huge night owl lol! what time is it where u are?", "i always stay up way too late scrolling on my phone :) what time is it for u?",
            "time zones are crazy on here haha! is it late where u live?", "haha i can never sleep early! what time is it over there?",
            "just chilling and losing track of time :) is it night or day where u are?", "yeah late night chats are always the best vibe :) what time is it there?",
            "haha my sleep schedule is a mess lol! what time is it for u right now?", "always up late texting and listening to music :) is it late where u are?",
        ), '["its late here","its midnight","cant sleep"]'),
        "s15_wthr": (43, "false", "weather_chat", (
            "the weather is pretty chill here today! hows the weather where u are?", "it's kinda nice and cozy outside here :) is it hot or cold where u live?",
            "pretty mild weather here right now! what's it like where u are?", "cozy weather for staying in bed haha :) is it warm or cold there?",
            "not bad at all here today! :) do u like hot summer weather or cold cozy weather?", "kinda chill outside so im staying cozy inside :) hows the weather there?",
            "it's pretty decent weather here :) what's it like in ur city?", "nice weather today! :) is it raining or sunny where u are?",
        ), '["its cold here","its hot here","pretty nice"]'),
    }


def compile_fsm_and_http_tables():
    specs = _pool_specs()
    pool_names = list(specs.keys())
    pid = {name: i for i, name in enumerate(pool_names)}
    n_pools = len(pool_names)

    blob_chunks = []
    cur_off = 0
    pool_off = np.zeros((n_pools, 32), dtype=np.int32)
    pool_end = np.zeros((n_pools, 32), dtype=np.int32)
    pool_mask = np.zeros((n_pools,), dtype=np.uint8)

    for p_idx, name in enumerate(pool_names):
        cid, is_g, int_s, msgs, kbd = specs[name]
        pool_mask[p_idx] = np.uint8(len(msgs) - 1)
        for j, m_txt in enumerate(msgs):
            raw_b = _pack_http(cid, is_g, int_s, m_txt, kbd)
            pool_off[p_idx, j] = np.int32(cur_off)
            cur_off += len(raw_b)
            pool_end[p_idx, j] = np.int32(cur_off)
            blob_chunks.append(raw_b)

    http_blob_u8 = np.frombuffer(b"".join(blob_chunks), dtype=np.uint8).copy()

    route_lut = np.zeros((NUM_LEAVES, 3, 3, 2), dtype=np.uint16)
    leaf_cls = np.zeros((NUM_LEAVES,), dtype=np.uint8)
    leaf_rule = np.zeros((NUM_LEAVES,), dtype=np.uint8)
    leaf_plock = np.zeros((NUM_LEAVES,), dtype=np.uint8)
    chk_lo = np.zeros((NUM_LEAVES,), dtype=np.uint32)
    chk_hi = np.zeros((NUM_LEAVES,), dtype=np.uint32)
    set_lo = np.zeros((NUM_LEAVES,), dtype=np.uint32)
    set_hi = np.zeros((NUM_LEAVES,), dtype=np.uint32)

    def bind(leaf, cid, p_norm, p_cmp=None, p_rep=None, p_chg=None, p_m=None, p_m_cmp=None, p_m_rep=None,
             chk=F_ZERO, set_f=F_GREETED, rule=RULE_STD, plock=0, p_part=None, p_m_part=None):
        leaf_cls[leaf] = np.uint8(cid)
        leaf_rule[leaf] = np.uint8(rule)
        leaf_plock[leaf] = np.uint8(plock)
        chk_lo[leaf] = chk
        set_lo[leaf] = set_f

        pf_0 = pid[p_norm]
        pf_p = pid[p_part] if p_part else pf_0
        pf_c = pid[p_cmp] if p_cmp else pf_0
        pf_r = pid[p_rep] if p_rep else pf_0
        pf_x = pid[p_chg] if p_chg else pf_r

        pm_0 = pid[p_m] if p_m else pf_0
        pm_p = pid[p_m_part] if p_m_part else (pid[p_part] if (p_part and not p_m) else pm_0)
        pm_c = pid[p_m_cmp] if p_m_cmp else (pid[p_cmp] if (p_cmp and not p_m) else pm_0)
        pm_r = pid[p_m_rep] if p_m_rep else (pid[p_rep] if (p_rep and not p_m) else pm_0)
        pm_x = pid[p_chg] if p_chg else pm_r

        route_lut[leaf, 0, 0, 1] = np.uint16(pf_0)
        route_lut[leaf, 0, 1, 1] = np.uint16(pf_p)
        route_lut[leaf, 0, 2, 1] = np.uint16(pf_c)
        route_lut[leaf, 0, 0, 0] = np.uint16(pm_0)
        route_lut[leaf, 0, 1, 0] = np.uint16(pm_p)
        route_lut[leaf, 0, 2, 0] = np.uint16(pm_c)
        route_lut[leaf, 1, :, 1] = np.uint16(pf_r)
        route_lut[leaf, 1, :, 0] = np.uint16(pm_r)
        route_lut[leaf, 2, :, 1] = np.uint16(pf_x)
        route_lut[leaf, 2, :, 0] = np.uint16(pm_x)

    bind(0, 0, "reject", set_f=F_ZERO)
    bind(1, 1, "greet_first", p_rep="greet_rep", chk=F_GREETED, set_f=F_GREETED, rule=RULE_GREET)
    bind(2, 2, "ask_age", p_rep="ask_age_rep", chk=F_ASKED_AGE, set_f=F_GREETED | F_ASKED_AGE)
    bind(3, 3, "tell_age", p_part="tell_age_cmp", p_cmp="tell_age_cmp", p_rep="tell_age_rep", p_chg="tell_age_chg", set_f=F_GREETED | F_ASKED_AGE, rule=RULE_AGE)
    bind(4, 4, "ask_ctry", p_rep="ask_ctry_rep", chk=F_ASKED_COUNTRY, set_f=F_GREETED | F_ASKED_COUNTRY)
    bind(5, 5, "tell_ctry", p_part="tell_ctry_cmp", p_cmp="tell_ctry_cmp", p_rep="tell_ctry_rep", p_chg="tell_ctry_chg", set_f=F_GREETED | F_ASKED_COUNTRY, rule=RULE_COUNTRY)
    bind(6, 6, "tell_age_ctry", set_f=F_GREETED | F_ASKED_AGE | F_ASKED_COUNTRY)
    bind(7, 7, "underage", set_f=F_ZERO)
    bind(8, 8, "ask_hru", set_f=F_GREETED | F_ASKED_HRU)
    # Leaf 9 checks F_ASKED_GENDER: if user already said "M"/"F" and sends "u", replies "im F :) how old r u?" (no double "u?")
    bind(9, 9, "g_choice_f", p_m="g_choice_m", p_rep="g_choice_f_known", p_m_rep="g_choice_m_known", chk=F_ASKED_GENDER, set_f=F_GREETED | F_ASKED_GENDER)
    bind(10, 10, "g_yes_f", p_m="g_no_m", set_f=F_GREETED | F_ASKED_GENDER)
    bind(11, 11, "g_no_f", p_m="g_yes_m", set_f=F_GREETED | F_ASKED_GENDER)
    bind(12, 12, "g_st_f_bf", p_m="g_st_f_bm", p_part="g_age_gen_f", p_m_part="g_age_gen_m", p_cmp="tell_age_cmp", set_f=F_GREETED | F_ASKED_GENDER, rule=RULE_SET_G_F)
    bind(13, 13, "g_st_m_bf", p_m="g_st_m_bm", p_part="g_age_gen_f", p_m_part="g_age_gen_m", p_cmp="tell_age_cmp", set_f=F_GREETED | F_ASKED_GENDER, rule=RULE_SET_G_M)
    bind(14, 14, "g_age_gen_f", p_m="g_age_gen_m", p_part="g_age_gen_f", p_m_part="g_age_gen_m", p_cmp="tell_age_cmp", set_f=F_GREETED | F_ASKED_GENDER, rule=RULE_SET_G_F)
    bind(15, 14, "g_age_gen_f", p_m="g_age_gen_m", p_part="g_age_gen_f", p_m_part="g_age_gen_m", p_cmp="tell_age_cmp", set_f=F_GREETED | F_ASKED_GENDER, rule=RULE_SET_G_M)
    bind(16, 15, "ask_wyd", p_rep="ask_wyd_rep", chk=F_ASKED_WYD, set_f=F_GREETED | F_ASKED_WYD)
    bind(17, 16, "st_pos", p_cmp="st_pos_cmp", set_f=F_GREETED | F_TOLD_STATUS)
    bind(18, 17, "st_neg", p_cmp="st_neg_cmp", set_f=F_GREETED | F_TOLD_STATUS)
    bind(19, 18, "st_ab", set_f=F_GREETED | F_TOLD_STATUS)
    bind(20, 19, "flirt", p_cmp="flirt_cmp", set_f=F_GREETED | F_FLIRTED)
    bind(21, 20, "react", p_cmp="react_cmp")
    bind(22, 21, "look_ask_f", p_m="look_ask_m", p_rep="look_ask_rep", chk=F_ASKED_LOOKING, set_f=F_GREETED | F_ASKED_LOOKING)
    bind(23, 22, "look_fr", p_cmp="look_fr_cmp", set_f=F_GREETED | F_STATED_LOOKING | F_ASKED_LOOKING)
    bind(24, 22, "look_g_bf", p_cmp="look_g_bf_cmp", p_m="look_g_bm", set_f=F_GREETED | F_STATED_LOOKING | F_ASKED_LOOKING)
    bind(25, 22, "look_b_bf", p_m="look_b_bm", set_f=F_GREETED | F_STATED_LOOKING | F_ASKED_LOOKING)
    bind(26, 22, "look_dat", p_cmp="look_dat_cmp", set_f=F_GREETED | F_STATED_LOOKING | F_ASKED_LOOKING)
    bind(27, 23, "name_ask_f", p_m="name_ask_m", p_rep="name_ask_f_rep", p_m_rep="name_ask_m_rep", chk=F_BOT_NAMED, set_f=F_GREETED | F_BOT_NAMED, plock=1)
    bind(28, 24, "name_tell_f", p_cmp="name_tell_f_cmp", p_m="name_tell_m", p_m_cmp="name_tell_m_cmp", p_rep="name_tell_rep", chk=F_USER_NAMED, set_f=F_GREETED | F_USER_NAMED | F_BOT_NAMED, plock=1)
    bind(29, 25, "hb_ask_mus")
    bind(30, 25, "hb_ask_gam")
    bind(31, 25, "hb_ask_mov")
    bind(32, 25, "hb_ask_spt")
    bind(33, 25, "hb_ask_gen")
    bind(34, 26, "hb_tell_mus")
    bind(35, 26, "hb_tell_gam")
    bind(36, 26, "hb_tell_mov")
    bind(37, 26, "hb_tell_spt")
    bind(38, 26, "hb_tell_gen")
    bind(39, 27, "rel_ask_f", p_m="rel_ask_m", p_rep="rel_ask_rep", chk=F_ASKED_REL, set_f=F_GREETED | F_ASKED_REL)
    bind(40, 28, "rel_single", p_cmp="rel_single_cmp", set_f=F_GREETED | F_ASKED_REL)
    bind(41, 28, "rel_taken", set_f=F_GREETED | F_ASKED_REL)
    bind(42, 28, "rel_married", set_f=F_GREETED | F_ASKED_REL)
    bind(43, 29, "priv_tg", p_cmp="priv_tg_cmp", p_rep="priv_rep", chk=F_ASKED_PRIV, set_f=F_GREETED | F_ASKED_PRIV)
    bind(44, 29, "priv_soc", p_cmp="priv_soc_cmp", p_rep="priv_rep", chk=F_ASKED_PRIV, set_f=F_GREETED | F_ASKED_PRIV)
    bind(45, 30, "yn_yes", p_cmp="yn_yes_cmp")
    bind(46, 30, "yn_no", p_cmp="yn_no_cmp")
    bind(47, 30, "yn_may", p_cmp="yn_may_cmp")
    bind(48, 31, "pic_ask", p_cmp="pic_ask_cmp", p_rep="pic_rep", chk=F_ASKED_PIC, set_f=F_GREETED | F_ASKED_PIC)
    bind(49, 31, "app_f", p_m="app_m")
    bind(50, 32, "pic_off")
    bind(51, 32, "pic_ref")
    bind(52, 33, "bot_no", p_cmp="bot_no_cmp", p_rep="bot_rep", chk=F_BOT_CHECKED, set_f=F_GREETED | F_BOT_CHECKED)
    bind(53, 34, "bot_real", p_cmp="bot_real_cmp", p_rep="bot_rep", chk=F_BOT_CHECKED, set_f=F_GREETED | F_BOT_CHECKED)
    bind(54, 35, "s15_bye")
    bind(55, 35, "s15_gn")
    bind(56, 35, "s15_brb")
    bind(57, 36, "s15_wk_ask")
    bind(58, 36, "s15_wk_tell")
    bind(59, 37, "s15_lang")
    bind(60, 37, "s15_badeng")
    bind(61, 38, "s15_ama")
    bind(62, 38, "s15_tod")
    bind(63, 38, "s15_bio_f", p_m="s15_bio_m")
    bind(64, 39, "s15_like")
    bind(65, 39, "s15_meet")
    bind(66, 39, "s15_aff")
    bind(67, 40, "s15_thx")
    bind(68, 40, "s15_sry")
    bind(69, 41, "s15_ping")
    bind(70, 41, "s15_clar")
    bind(71, 42, "s15_nsfw")
    bind(72, 42, "s15_rude")
    bind(73, 43, "s15_time")
    bind(74, 43, "s15_wthr")
    bind(75, 99, "reset", set_f=F_ZERO)
    # Leaf 76: Standalone Context Ask-Back ("u", "u?", "and u?", "wbu?", "hbu?") -> redirected via askback_redirect_u8!
    bind(76, 0, "reject", set_f=F_ZERO)
    # Leaf 77: "Chatting with u" / "talking with u" / "texting u" (Class 16)
    bind(77, 16, "st_chat_u", p_cmp="st_chat_u_cmp", set_f=F_GREETED | F_TOLD_STATUS)

    # --- O(1) CONTEXT-AWARE ASK-BACK REDIRECT TABLE (`askback_redirect_u8`, 78 bytes total!) ---
    # When user sends "u" / "u?" / "and u?" / "wbu?" as a separate message:
    # - After "Chicken" (last_leaf=0) or start of chat (0) -> redirects to 0 (Class 0 Reject! Zero hallucination!)
    # - After "M" (13) or "F" (12) -> redirects to 9 (ask_gender_choice -> "im F :) how old r u?")
    # - After "24" (3) -> redirects to 2 (ask_age -> "im 22! where r u from?")
    # - After "Chatting with u" (77) or "im good" (17) -> redirects to 19 (status_ask_back -> "I'm good! just chilling in bed :)")
    askback_redirect = np.zeros((NUM_LEAVES,), dtype=np.uint8)
    for prev_lf, target_lf in (
        (3, 2), (5, 4), (6, 2),
        (12, 9), (13, 9), (14, 9), (15, 9),
        (17, 19), (18, 19), (77, 19),
        (20, 20),
        (23, 22), (24, 22), (25, 22), (26, 22),
        (28, 27),
        (34, 29), (35, 30), (36, 31), (37, 32), (38, 33),
        (40, 39), (41, 39), (42, 39),
        (58, 57), (59, 59), (60, 59),
        (73, 73), (74, 74),
    ):
        askback_redirect[prev_lf] = np.uint8(target_lf)

    return {
        "http_blob_u8": http_blob_u8,
        "pool_msg_off_i32": pool_off,
        "pool_msg_end_i32": pool_end,
        "pool_mask_u8": pool_mask,
        "route_pool_lut": route_lut,
        "leaf_to_class_u8": leaf_cls,
        "leaf_entity_rule_u8": leaf_rule,
        "leaf_persona_lock_u8": leaf_plock,
        "leaf_chk_lo_u32": chk_lo,
        "leaf_chk_hi_u32": chk_hi,
        "leaf_set_lo_u32": set_lo,
        "leaf_set_hi_u32": set_hi,
        "askback_redirect_u8": askback_redirect,
    }


# --- RULE 14 ZERO-FAULT ADVERSARIAL REGRESSION SUITE (STEPS 1 TO 15 + NEW EDGE CASES) ---
ZERO_FAULT_SUITE = (
    # Step 1: Greetings (Class 1)
    (b"hi", 1, 1), (b"hi hi", 1, 1), (b"hy", 1, 1), (b"hyy", 1, 1), (b"hiiii", 1, 1),
    (b"hey", 1, 1), (b"hey hey", 1, 1), (b"heeyy", 1, 1), (b"hello", 1, 1), (b"helloo", 1, 1),
    (b"hlo", 1, 1), (b"yo", 1, 1), (b"sup", 1, 1), (b"good morning", 1, 1), (b"whats up", 1, 1),
    (b"hi there", 1, 1), (b"hey man", 1, 1), (b"nice to meet u", 1, 1), (b"hey cutie", 1, 1),
    # Step 2: Age, Country, Combined, Underage, Reset (Classes 2-7, 99)
    (b"age?", 2, 2), (b"how old r u", 2, 2), (b"how old are you", 2, 2),
    (b"24", 3, 3), (b"im 24", 3, 3), (b"lm 24", 3, 3), (b"24y", 3, 3), (b"21yres", 3, 3), (b"im24", 3, 3),
    (b"from?", 4, 4), (b"where r u from", 4, 4), (b"where are you from", 4, 4),
    (b"tuinsi", 5, 5), (b"handi", 5, 5), (b"ind", 5, 5), (b"tunisia", 5, 5), (b"im from usa", 5, 5),
    (b"hi l'm 30 from tun", 6, 6), (b"24 from uk", 6, 6),
    (b"15", 7, 7), (b"im 16", 7, 7), (b"next", 99, 75), (b"/start", 99, 75),
    # Step 3: Ask HRU (Class 8)
    (b"how are you", 8, 8), (b"how are you?", 8, 8), (b"how are you!", 8, 8), (b"hru", 8, 8),
    (b"hyd", 8, 8), (b"how r u", 8, 8), (b"how u doing!", 8, 8), (b"how was your day", 8, 8),
    (b"hor", 8, 8), (b"how ar you dopin", 8, 8), (b"hw r u", 8, 8), (b"hi how r u", 8, 8),
    (b"hello how are you doing!", 8, 8),
    # Step 4: Gender (Classes 9-14)
    (b"m or f", 9, 9), (b"m or f?", 9, 9), (b"m/f", 9, 9), (b"f/m", 9, 9), (b"gender?", 9, 9), (b"what gender are you", 9, 9),
    (b"f?", 10, 10), (b"girl?", 10, 10), (b"are you a girl", 10, 10), (b"r u female", 10, 10), (b"u f?", 10, 10),
    (b"m?", 11, 11), (b"boy?", 11, 11), (b"are you a boy", 11, 11), (b"r u male", 11, 11), (b"u m?", 11, 11),
    (b"f", 12, 12), (b"famle", 12, 12), (b"f here", 12, 12), (b"im a girl", 12, 12), (b"hi f", 12, 12), (b"lm f u", 12, 12), (b"im f u?", 12, 12),
    (b"m", 13, 13), (b"male", 13, 13), (b"m here", 13, 13), (b"im a boy", 13, 13), (b"hi m", 13, 13), (b"im m u?", 13, 13),
    (b"22f", 14, 14), (b"24 m", 14, 15), (b"19 famle", 14, 14), (b"21 girl", 14, 14),
    # Step 5: Ask WYD (Class 15)
    (b"wyd", 15, 16), (b"wyd?", 15, 16), (b"what are you doing", 15, 16), (b"what r u doing", 15, 16), (b"what u up to", 15, 16),
    # Step 6: Status + Dedicated "Chatting with u" (Classes 16-18)
    (b"im good", 16, 17), (b"im ok", 16, 17), (b"just chilling in bed", 16, 17), (b"just chatting", 16, 17), (b"listening to music", 16, 17),
    (b"Chatting with u", 16, 77), (b"chatting with you", 16, 77), (b"talking with u", 16, 77), (b"texting u", 16, 77), (b"im chatting with u", 16, 77),
    (b"bored", 17, 18), (b"lm borad", 17, 18), (b"nothing much", 17, 18), (b"im tired", 17, 18),
    (b"im good u?", 18, 19), (b"lm borad u?", 18, 19), (b"nmu", 18, 19), (b"nothing much wbu?", 18, 19),
    # Step 7: Flirt & Reactions (Classes 19-20)
    (b"ur cute", 19, 20), (b"u seem nice", 19, 20), (b"i like u", 19, 20), (b"u look good", 19, 20), (b"love ur vibe", 19, 20),
    (b"haha", 20, 21), (b"lol", 20, 21), (b"nice", 20, 21), (b"cool", 20, 21), (b"true", 20, 21), (b"ok", 20, 21),
    (b"same", 20, 21), (b"sam", 20, 21), (b"same here", 20, 21), (b"sam here", 20, 21), (b"for real", 20, 21),
    # Step 8: Looking For (Classes 21-22)
    (b"what r u looking for?", 21, 22), (b"why r u on here?", 21, 22), (b"looking for friends or dating?", 21, 22), (b"u looking for a girl?", 21, 22),
    (b"looking for friends", 22, 23), (b"making friends", 22, 23), (b"friends", 22, 23),
    (b"m looking for girl , u", 22, 24), (b"lm finding gril", 22, 24), (b"im looking for a gf", 22, 24), (b"a gf", 22, 24),
    (b"f looking for boy wbu?", 22, 25), (b"a bf", 22, 25),
    (b"looking for dating", 22, 26), (b"a relationship", 22, 26),
    # Step 9: Name (Classes 23-24)
    (b"whats ur name?", 23, 27), (b"what is your name", 23, 27), (b"ur name?", 23, 27), (b"name?", 23, 27), (b"what should i call u?", 23, 27),
    (b"hatem", 24, 28), (b"im hatem", 24, 28), (b"my name is sarah", 24, 28), (b"call me alex", 24, 28), (b"im sam", 24, 28), (b"my name is zoltan", 24, 28),
    # Step 10: Hobbies (Classes 25-26)
    (b"what music do u like?", 25, 29), (b"what games do u play?", 25, 30), (b"do u watch anime?", 25, 31),
    (b"do u go to the gym?", 25, 32), (b"what r ur hobbies?", 25, 33), (b"what do u do for fun?", 25, 33),
    (b"i like rap music", 26, 34), (b"i play valorant", 26, 35), (b"i love anime", 26, 36), (b"going to the gym", 26, 37), (b"i love traveling", 26, 38),
    # Step 11: Relationship Status (Classes 27-28)
    (b"are you single?", 27, 39), (b"are you single or mariid", 27, 39), (b"do u have a bf?", 27, 39), (b"are u in a relationship?", 27, 39),
    (b"im single", 28, 40), (b"im single u?", 28, 40), (b"im not married", 28, 40), (b"i dont have a gf", 28, 40),
    (b"im taken", 28, 41), (b"im in a relationship", 28, 41), (b"im married", 28, 42), (b"lm mariid", 28, 42),
    # Step 12: Private / Telegram / Socials & Yes/No/Maybe (Classes 29-30)
    (b"can we talk privty", 29, 43), (b"can we talk private?", 29, 43), (b"ur telegram?", 29, 43), (b"telgrm?", 29, 43), (b"dm me", 29, 43),
    (b"do u have snap?", 29, 44), (b"ur insta?", 29, 44), (b"whatsapp?", 29, 44),
    (b"yes", 30, 45), (b"yes plz", 30, 45), (b"no", 30, 46), (b"nope", 30, 46), (b"i dont have snap", 30, 46), (b"maybe", 30, 47), (b"idk", 30, 47),
    # Step 13: Pic / Photo / Appearance (Classes 31-32)
    (b"send pic", 31, 48), (b"send a selfie", 31, 48), (b"pic for pic", 31, 48), (b"can i see u?", 31, 48),
    (b"what do u look like?", 31, 49), (b"how tall r u?", 31, 49),
    (b"wanna see me?", 32, 50), (b"can i send a pic?", 32, 50), (b"u first", 32, 51), (b"i dont send pics", 32, 51),
    # Step 14: Bot / Real Check (Classes 33-34)
    (b"are u a bot?", 33, 52), (b"bot?", 33, 52), (b"are u ai?", 33, 52), (b"bot or real?", 33, 52), (b"u reply so fast", 33, 52),
    (b"are u real?", 34, 53), (b"are u real", 34, 53), (b"are u human?", 34, 53), (b"prove ur real", 34, 53),
    # Step 15: Unified Dating Pack (Classes 35-43)
    (b"bye", 35, 54), (b"gtg", 35, 54), (b"goodnight", 35, 55), (b"gn", 35, 55), (b"brb", 35, 56), (b"wait a sec", 35, 56),
    (b"do u work or study?", 36, 57), (b"are u a student?", 36, 57), (b"im a student", 36, 58), (b"in college", 36, 58),
    (b"do u speak arabic?", 37, 59), (b"do u speak french?", 37, 59), (b"my english is bad", 37, 60),
    (b"ask me anything", 38, 61), (b"truth or dare", 38, 62), (b"tell me about urself", 38, 63),
    (b"do u like me?", 39, 64), (b"can we meet?", 39, 65), (b"i miss u", 39, 66), (b"send a hug", 39, 66),
    (b"thanks", 40, 67), (b"thank u", 40, 67), (b"sorry", 40, 68), (b"my bad", 40, 68),
    (b"u there?", 41, 69), (b"still there?", 41, 69), (b"wdym", 41, 70), (b"what do u mean?", 41, 70),
    (b"send nudes", 42, 71), (b"wanna sext", 42, 71), (b"shut up", 42, 72), (b"fuck u", 42, 72),
    (b"what time is it there?", 43, 73), (b"why r u still awake?", 43, 73), (b"hows the weather", 43, 74),
    # Context Ask-Back Triggers (Leaf 76 -> redirected via askback_redirect_u8[prev_leaf] in production!)
    (b"u", 0, 76), (b"u?", 0, 76), (b"and u?", 0, 76), (b"wbu?", 0, 76), (b"what about u?", 0, 76),
    # Class 0 Traps (Includes "Chicken"!)
    (b"Chicken", 0, 0), (b"what", 0, 0), (b"what?", 0, 0), (b"what!", 0, 0), (b"what do you do", 0, 0), (b"what is this", 0, 0),
    (b"why", 0, 0), (b"why?", 0, 0), (b"why not", 0, 0), (b"who", 0, 0), (b"who?", 0, 0), (b"who are you", 0, 0),
    (b"hyw", 0, 0), (b"iphey", 0, 0), (b"m bi", 0, 0), (b"a bf here", 0, 0), (b"a gf here", 0, 0), (b"asdfgh", 0, 0),
)


if __name__ == "__main__":
    compiled = compile_fsm_and_http_tables()
    tot_kb = sum(arr.nbytes for arr in compiled.values()) / 1024.0
    print(
        f"[OFFLINE FSM COMPILER OK] Compiled {NUM_LEAVES} leaf intents, {len(compiled['pool_mask_u8'])} pools, "
        f"{len(ZERO_FAULT_SUITE)} Zero-Fault tests -> {tot_kb:.2f} KB total binary blob!",
        flush=True,
    )
