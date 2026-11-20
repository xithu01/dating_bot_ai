#!/usr/bin/env python3
# FILE: offline_spec_01_core.py
# PURPOSE: Rule 15 Offline Spec Module 1 — Steps 1-7 (Classes 0-20, 99) + Context Ask-Back (76) + "Chatting with u" (77)
# RULES: Enforces Rules 1-15 (<= 220 lines, auto-discovered by offline_fsm_compiler.py, zero float64/int64)

import re
import numpy as np

from offline_fsm_compiler import (
    F_ASKED_AGE, F_ASKED_COUNTRY, F_ASKED_GENDER, F_ASKED_HRU, F_ASKED_WYD,
    F_FLIRTED, F_GREETED, F_TOLD_STATUS, F_ZERO,
    RULE_AGE, RULE_COUNTRY, RULE_GREET, RULE_SET_G_F, RULE_SET_G_M,
)

POOLS = {
    "reject": (0, "false", "not_valid", ("say hi, how r u, M or F, ur age (18+), or where u are from!",), '["hey","how r u?","im 22"]'),
    "underage": (7, "false", "underage_reject", ("sorry, 18+ adults only!",), '["Exit"]'),
    "reset": (99, "false", "system_reset", ("new chat started! say hi :)",), '["hey","hi!"]'),
    "greet_first": (1, "true", "english_greeting", (
        "hey", "hi", "heyy", "yo", "hey there", "hii", "sup", "hello!", "hey how r u", "hi! hru?", "yo whats up", "heyy :)", "hiya", "hey! hows it going", "hiii", "sup! hru",
        "hello there", "hey u", "hi there!", "yo! how are u", "heyyy", "whats up!", "hey! nice to meet u", "hi :) how r u", "howdy!", "hey whats up", "hii there", "hello :)", "yo! whats good", "hey! hru doing", "hi! hows u", "heyy! whats up",
    ), '["good u?","how r u?"]'),
    "greet_rep": (1, "true", "english_greeting_repeat", (
        "hey again lol", "haha hi again!", "we already said hi :)", "yo again! how r u?", "hii again haha", "already said hey lol", "haha hey x2! hru?", "hi again! whats up?",
    ), '["haha im good","how r u?"]'),
    "ask_age": (2, "false", "ask_age", ("22 u?", "im 23, how old r u?", "22! wbu?", "im 23 :)", "22 here, u?", "23! how old are u", "im 22, ur age?", "23 u?"), '["im 21","im 24","im 28"]'),
    "ask_age_rep": (2, "false", "ask_age_repeat", ("im 22! where r u from?", "im 23 :) hows ur day going?", "22 here! what country u from?", "im 22 haha :) wyd rn?", "23 here! how r u doing?", "im 22 :) what u up to today?", "22 here! where u live?", "im 23 :) whats up?"), '["from usa","im good"]'),
    "tell_age": (3, "false", "tell_age", ("nice! im 22, where u from?", "cool im 23! u from?", "nice age! im 22, from?", "22 here! where r u from?", "ooh nice, im 23! country?", "cool! 22 here, u from?", "nice! im 23, where u live?", "sweet, im 22! where u from?"), '["usa","uk","tunisia"]'),
    "tell_age_cmp": (3, "false", "tell_age_complete", ("nice! im 22 too :) hows ur day going?", "cool age! im 23, what u up to today?", "sweet! 22 here :) how r u doing?", "ooh nice! im 23, hows it going?", "nice! 22 here, what u doing rn?", "cool! im 23 :) having a good day?", "awesome! im 22, whats up?", "nice age! 23 here, hows ur day?"), '["going great!","chilling u?"]'),
    "tell_age_rep": (3, "false", "tell_age_repeat", ("haha u already said ur age! where r u from?", "ik ur age lol! what country u in?", "u told me ur age already haha! u from?", "haha yeah ik! where u from?", "already got ur age lol :) where u live?", "haha u said that! what country?", "ik lol! where r u from?", "yeah u told me haha! u from?"), '["usa","uk","tunisia"]'),
    "tell_age_chg": (3, "false", "tell_age_changed", ("wait u said a different age before lol! where u from?", "haha changing ur age now? lol how r u", "wait didnt u say another age earlier lol?", "haha u changed ur age! which one is real?", "hold on u said a different age before lol!", "haha new age unlocked lol! whats up?", "wait u changed ur age haha!", "lol u said another age before! how r u?"), '["haha jk","lol my bad"]'),
    "ask_ctry": (4, "false", "ask_country", ("from usa, u?", "uk! where u from?", "im from usa :)", "canada, wbu?", "usa here! u from?", "from uk, where r u from?", "usa! what about u?", "im in usa, u?"), '["nice!","me too","cool"]'),
    "ask_ctry_rep": (4, "false", "ask_country_repeat", ("im from usa :) how old r u?", "im from uk! hows ur day going?", "usa here :) what u up to rn?", "from usa :) how r u doing?", "im from usa! what r ur hobbies?", "usa here :) having a good day?", "im from uk :) whats up?", "from usa :) wyd rn?"), '["nice!","im good"]'),
    "tell_ctry": (5, "false", "tell_country", ("ooh nice! im from usa, how old r u?", "cool place! im from uk, age?", "nice! im from usa :) ur age?", "love that! im from usa, how old r u?", "awesome! usa here, how old u?", "nice country! im from uk, age?", "cool! im from canada, how old r u?", "sweet! usa here, ur age?"), '["im 21","im 24","im 28"]'),
    "tell_ctry_cmp": (5, "false", "tell_country_complete", ("ooh nice! im from usa :) hows ur day going?", "cool place! im from uk, what u up to?", "love that! usa here :) how r u doing?", "awesome place! im from canada, whats up?", "nice! im from usa, what u doing rn?", "sweet! uk here :) having a good day?", "ooh cool! im from usa, hows it going?", "nice country! usa here, hows ur day?"), '["going great!","chilling u?"]'),
    "tell_ctry_rep": (5, "false", "tell_country_repeat", ("ik ur from there lol! how old r u?", "haha u already told me ur country! whats up?", "yeah u said that already lol! how r u?", "ik lol! nice place :) what u up to?", "haha already got ur country! how old r u?", "u told me already lol! hows it going?", "yeah ik haha! whats up?", "haha u said that! how r u doing?"), '["haha im good","chilling"]'),
    "tell_ctry_chg": (5, "false", "tell_country_changed", ("wait u said a different country before lol!", "haha moving countries fast lol! how r u?", "wait didnt u say another country earlier lol?", "haha traveling the world? u said another place before!", "hold on u changed ur country lol!", "haha new country unlocked! whats up?", "wait u said somewhere else before haha!", "lol changing countries now? how r u?"), '["haha jk","traveling lol"]'),
    "tell_age_ctry": (6, "true", "tell_age_and_country", ("hey! nice im 22 from usa, how r u?", "hi! cool im 23 from uk, hru?", "nice to meet u! 22 from usa here :)", "heyy! im 22 from canada, hows it going?", "yo! 23 from usa here, whats up?", "hey there! im 22 from uk, how r u?", "hi! 22 from usa here, nice to meet u!", "heyy! cool, im 23 from usa, hru?"), '["im good u?","doing great!"]'),
    "ask_hru": (8, "false", "ask_how_are_you", ("im doing great! how old r u and where u from?", "doing good thanks! wbu, what's ur age and country?", "im good! im 22 from usa, how old r u and where u from?", "all good here! how old are u and where r u from?", "pretty good! im 23 from uk, ur age and country?", "doing great :) how old r u and where u live?", "im great thanks! wbu, age and where u from?", "good! chilling here, how old r u and what country?"), '["im 22","from usa","24 from uk"]'),
    "g_choice_f": (9, "false", "ask_gender_choice", ("im F :) u?", "im a girl! wbu?", "female here, u?", "F! what about u?", "im a girl :) u M or F?", "F here! wbu?", "im female, how about u?", "girl here :) u?"), '["im M","im F"]'),
    "g_choice_m": (9, "false", "ask_gender_choice", ("im M :) u?", "im a guy! wbu?", "male here, u?", "M! what about u?", "im a boy :) u M or F?", "M here! wbu?", "im male, how about u?", "guy here :) u?"), '["im F","im M"]'),
    "g_choice_f_known": (9, "false", "ask_gender_choice_answer", ("im F :) how old r u?", "im a girl! where r u from?", "female here :) whats ur age and country?", "im a girl, 22 :) hows ur day going?", "F here! how old r u?", "girl here :) what country u from?", "im female, 22 :) what u up to rn?", "im a girl :) how r u doing?"), '["im 22","from usa","wyd?"]'),
    "g_choice_m_known": (9, "false", "ask_gender_choice_answer", ("im M :) how old r u?", "im a guy! where r u from?", "male here :) whats ur age and country?", "im a guy, 23 :) hows ur day going?", "M here! how old r u?", "guy here :) what country u from?", "im male, 23 :) what u up to rn?", "im a guy :) how r u doing?"), '["im 22","from usa","wyd?"]'),
    "g_yes_f": (10, "false", "ask_gender_is_female", ("yes im a girl! u?", "yep F here! wbu?", "yes! im female :) u?", "yeah im a girl, how old r u?", "yes F here! where u from?", "yep im a girl :) wbu?", "yes! female here, u M or F?", "yeah F! what about u?"), '["im M","im F","im 22"]'),
    "g_no_m": (10, "false", "ask_gender_is_female", ("no im a guy actually! u?", "nope M here! wbu?", "no, im male :) u?", "nope im a boy lol, wbu?", "no im M! how old r u?", "nope, guy here :) u?", "no im male! where u from?", "nope M! what about u?"), '["im F","im M","im 22"]'),
    "g_no_f": (11, "false", "ask_gender_is_male", ("no im a girl actually! u?", "nope F here! wbu?", "no, im female :) u?", "nope im a girl lol, wbu?", "no im F! how old r u?", "nope, girl here :) u?", "no im female! where u from?", "nope F! what about u?"), '["im M","im F","im 22"]'),
    "g_yes_m": (11, "false", "ask_gender_is_male", ("yes im a guy! u?", "yep M here! wbu?", "yes! im male :) u?", "yeah im a boy, how old r u?", "yes M here! where u from?", "yep im a guy :) wbu?", "yes! male here, u M or F?", "yeah M! what about u?"), '["im F","im M","im 22"]'),
    "g_st_f_bf": (12, "false", "state_gender_female", ("im F too! how old r u?", "girl here too! ur age?", "same, female here :) how old r u?", "im a girl too! age and where u from?", "F too! how old r u?", "same here :) what country u from?", "im F too, im 22, how old r u?", "girl here too! how old are u?"), '["im 21","im 24","from usa"]'),
    "g_st_f_bm": (12, "false", "state_gender_female", ("im M! how old r u?", "guy here! ur age?", "male here :) how old r u?", "im a guy! age and where u from?", "M here! how old r u?", "im a boy :) what country u from?", "im M, im 23, how old r u?", "guy here! how old are u?"), '["im 21","im 24","from usa"]'),
    "g_st_m_bf": (13, "false", "state_gender_male", ("im F! how old r u?", "girl here! ur age?", "female here :) how old r u?", "im a girl! age and where u from?", "F here! how old r u?", "im a girl :) what country u from?", "im F, im 22, how old r u?", "girl here! how old are u?"), '["im 21","im 24","from usa"]'),
    "g_st_m_bm": (13, "false", "state_gender_male", ("im M too! how old r u?", "guy here too! ur age?", "same, male here :) how old r u?", "im a guy too! age and where u from?", "M too! how old r u?", "same here :) what country u from?", "im M too, im 23, how old r u?", "guy here too! how old are u?"), '["im 21","im 24","from usa"]'),
    "g_age_gen_f": (14, "false", "state_age_and_gender", ("nice! im F, 22, where r u from?", "cool! 22 F here, what country u from?", "sweet! im a girl, 22, where u from?", "ooh nice! F 23 here, where u live?", "awesome! im F 22, u from?", "nice! girl here, 22, where r u from?", "cool :) im F 22, what country?", "sweet! 22 F here, where u from?"), '["usa","uk","tunisia"]'),
    "g_age_gen_m": (14, "false", "state_age_and_gender", ("nice! im M, 23, where r u from?", "cool! 23 M here, what country u from?", "sweet! im a guy, 23, where u from?", "ooh nice! M 23 here, where u live?", "awesome! im M 23, u from?", "nice! guy here, 23, where r u from?", "cool :) im M 23, what country?", "sweet! 23 M here, where u from?"), '["usa","uk","tunisia"]'),
    "ask_wyd": (15, "false", "ask_wyd", ("just chilling in bed, wbu?", "nothing much, just bored lol, u?", "doing good! just relaxing, what u up to?", "just chilling listening to music, wbu?", "not much, just laying in bed :) u?", "just scrolling on my phone lol, wbu?", "chilling at home, what are u doing?", "nothing much, just bored! what u up to?"), '["chilling too","nothing much","in bed"]'),
    "ask_wyd_rep": (15, "false", "ask_wyd_repeat", ("haha still just chilling in bed! u?", "still bored lol, nothing changed! wbu?", "haha told u, just relaxing! what u up to?", "still laying in bed lol :) u?", "haha same thing, just chilling! wbu?", "still doing nothing much lol! u?", "haha still scrolling on my phone, wbu?", "still chilling at home :) what about u?"), '["haha nice","same here"]'),
    "st_pos": (16, "false", "status_positive", ("nice! :) how old r u and where u from?", "ok nice! what's ur age and country?", "cool! :) how old r u?", "nice! im chilling too, where r u from?", "awesome! how old are u and where u from?", "ok cool :) ur age and country?", "nice! im 22, how old r u?", "sweet! :) where r u from?"), '["im 22","from usa","24 m"]'),
    "st_pos_cmp": (16, "false", "status_positive_complete", ("nice! :) sounds chill", "ok cool! same here, just relaxing :)", "nice! im just chilling in bed too", "awesome :) glad ur doing good!", "ok nice! chilling here too", "cool! :) having a good day then", "nice :) im just laying in bed", "sweet! :) just relaxing here too"), '["wyd?","haha yeah"]'),
    "st_chat_u": (16, "false", "status_chatting_with_u", ("aww haha same, just chilling in bed chatting with u! :) how old r u?", "haha same here, texting u right now :) where r u from?", "aww sweet! talking to u too :) what's ur age and country?", "haha nice! im relaxing and chatting with u too :) how old r u?", "aww glad ur enjoying talking to me! :) where r u from?", "haha same! laying in bed texting u :) how old are u and where u from?", "aww thats cute, chatting with u too! :) what country u in?", "haha me too, just chilling and texting u :) how old r u?"), '["im 22","from usa","ur cute"]'),
    "st_chat_u_cmp": (16, "false", "status_chatting_with_u_complete", ("aww haha same, chatting with u is super fun! :) hows ur day going?", "haha same here, just chilling in bed texting u :) what r ur hobbies?", "aww sweet! talking to u too :) u have a really cool vibe!", "haha nice! im relaxing and texting u right now too :) what music do u like?", "aww thats cute :) im enjoying chatting with u too!", "haha same! best part of my day so far :) what do u like doing for fun?", "aww glad we matched :) im chilling and texting u too!", "haha me too! u are really fun to talk to :)"), '["haha thanks","ur cute too","what music u like?"]'),
    "st_neg": (17, "false", "status_negative", ("ok haha same, im bored too! how old r u?", "aw why bored? where r u from?", "haha same, nothing much here! ur age and country?", "ok lol let's chat then! how old r u?", "same here lol, just bored! where u from?", "haha i feel u! how old r u and where u from?", "aw hope u feel better! what's ur age?", "haha bored too! how old are u?"), '["im 22","from usa","24 m"]'),
    "st_neg_cmp": (17, "false", "status_negative_complete", ("haha same, im bored in bed too!", "aw why so bored? let's chat :)", "ok lol same here, nothing much going on", "haha i feel u, today is slow!", "same lol, just laying in bed bored", "aw let's keep each other company then :)", "haha nothing much here either!", "ok lol we are both bored then :)"), '["haha true","wyd?"]'),
    "st_ab": (18, "false", "status_ask_back", ("I'm good! just chilling in bed :) how old r u?", "Just chilling in bed! where r u from?", "Nothing much, just bored lol! ur age and country?", "Doing good! just relaxing :) how old are u?", "im good too! just chilling, where u from?", "just laying in bed bored lol! how old r u?", "doing good thanks! what country u from?", "nothing much, just chilling here :) ur age?"), '["im 22","from usa","24 m"]'),
    "flirt": (19, "false", "compliment_flirt", ("aw thank u! ur sweet :) how old r u?", "haha stop ur making me blush! where r u from?", "aw thanks cutie! how old r u and where u from?", "thank u :) u seem really cool too! ur age?", "aw thats so sweet! where r u from?", "haha thank u! i like talking to u too, how old r u?", "aw ur making me smile :) what country u from?", "thanks cutie :) how old r u and where u from?"), '["im 22","from usa","24 m"]'),
    "flirt_cmp": (19, "false", "compliment_flirt_complete", ("aw thank u! ur making me blush :)", "haha thanks cutie! u seem really sweet too", "aw i like talking to u too! what u up to?", "thank u :) ur pretty cute yourself!", "aw thats so sweet of u to say :)", "haha stop ur making me smile! wyd rn?", "thanks cutie :) love ur vibe too!", "aw thank u! ur my type too haha :)"), '["haha thanks","wyd?","ur sweet"]'),
    "react": (20, "false", "short_reaction_ack", ("haha yeah! btw how old r u?", "ik right :) where r u from?", "haha for real! what's ur age and country?", "glad we agree :) how old r u?", "haha ikr! where u from btw?", "sweet :) how old are u and where u from?", "haha true! what country u in?", "nice :) btw how old r u?"), '["im 22","from usa","24 m"]'),
    "react_cmp": (20, "false", "short_reaction_ack_complete", ("haha ikr! so what u up to rn?", "glad we vibe :) what do u like doing for fun?", "haha for real! having a good day?", "ikr :) ur fun to talk to!", "haha yeah! chilling in bed here, wbu?", "totally :) love ur vibe!", "haha true! what r u up to today?", "nice :) u seem really chill!"), '["just chilling","wyd?","thanks!"]'),
}

LEAF_BINDINGS = (
    (0, 0, "reject", {"set_f": F_ZERO}),
    (1, 1, "greet_first", {"p_rep": "greet_rep", "chk": F_GREETED, "set_f": F_GREETED, "rule": RULE_GREET}),
    (2, 2, "ask_age", {"p_rep": "ask_age_rep", "chk": F_ASKED_AGE, "set_f": F_GREETED | F_ASKED_AGE}),
    (3, 3, "tell_age", {"p_part": "tell_age_cmp", "p_cmp": "tell_age_cmp", "p_rep": "tell_age_rep", "p_chg": "tell_age_chg", "set_f": F_GREETED | F_ASKED_AGE, "rule": RULE_AGE}),
    (4, 4, "ask_ctry", {"p_rep": "ask_ctry_rep", "chk": F_ASKED_COUNTRY, "set_f": F_GREETED | F_ASKED_COUNTRY}),
    (5, 5, "tell_ctry", {"p_part": "tell_ctry_cmp", "p_cmp": "tell_ctry_cmp", "p_rep": "tell_ctry_rep", "p_chg": "tell_ctry_chg", "set_f": F_GREETED | F_ASKED_COUNTRY, "rule": RULE_COUNTRY}),
    (6, 6, "tell_age_ctry", {"set_f": F_GREETED | F_ASKED_AGE | F_ASKED_COUNTRY}),
    (7, 7, "underage", {"set_f": F_ZERO}),
    (8, 8, "ask_hru", {"set_f": F_GREETED | F_ASKED_HRU}),
    (9, 9, "g_choice_f", {"p_m": "g_choice_m", "p_rep": "g_choice_f_known", "p_m_rep": "g_choice_m_known", "chk": F_ASKED_GENDER, "set_f": F_GREETED | F_ASKED_GENDER}),
    (10, 10, "g_yes_f", {"p_m": "g_no_m", "set_f": F_GREETED | F_ASKED_GENDER}),
    (11, 11, "g_no_f", {"p_m": "g_yes_m", "set_f": F_GREETED | F_ASKED_GENDER}),
    (12, 12, "g_st_f_bf", {"p_m": "g_st_f_bm", "p_part": "g_age_gen_f", "p_m_part": "g_age_gen_m", "p_cmp": "tell_age_cmp", "set_f": F_GREETED | F_ASKED_GENDER, "rule": RULE_SET_G_F}),
    (13, 13, "g_st_m_bf", {"p_m": "g_st_m_bm", "p_part": "g_age_gen_f", "p_m_part": "g_age_gen_m", "p_cmp": "tell_age_cmp", "set_f": F_GREETED | F_ASKED_GENDER, "rule": RULE_SET_G_M}),
    (14, 14, "g_age_gen_f", {"p_m": "g_age_gen_m", "p_part": "g_age_gen_f", "p_m_part": "g_age_gen_m", "p_cmp": "tell_age_cmp", "set_f": F_GREETED | F_ASKED_GENDER, "rule": RULE_SET_G_F}),
    (15, 14, "g_age_gen_f", {"p_m": "g_age_gen_m", "p_part": "g_age_gen_f", "p_m_part": "g_age_gen_m", "p_cmp": "tell_age_cmp", "set_f": F_GREETED | F_ASKED_GENDER, "rule": RULE_SET_G_M}),
    (16, 15, "ask_wyd", {"p_rep": "ask_wyd_rep", "chk": F_ASKED_WYD, "set_f": F_GREETED | F_ASKED_WYD}),
    (17, 16, "st_pos", {"p_cmp": "st_pos_cmp", "set_f": F_GREETED | F_TOLD_STATUS}),
    (18, 17, "st_neg", {"p_cmp": "st_neg_cmp", "set_f": F_GREETED | F_TOLD_STATUS}),
    (19, 18, "st_ab", {"set_f": F_GREETED | F_TOLD_STATUS}),
    (20, 19, "flirt", {"p_cmp": "flirt_cmp", "set_f": F_GREETED | F_FLIRTED}),
    (21, 20, "react", {"p_cmp": "react_cmp"}),
    (75, 99, "reset", {"set_f": F_ZERO}),
    (76, 0, "reject", {"set_f": F_ZERO}),
    (77, 16, "st_chat_u", {"p_cmp": "st_chat_u_cmp", "set_f": F_GREETED | F_TOLD_STATUS}),
)

LEAF_STEP_BITS = {
    1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4,
    16: 5, 17: 6, 18: 6, 19: 6, 20: 7, 21: 7, 76: 16, 77: 6,
}

ASKBACK_REDIRECTS = (
    (3, 2), (5, 4), (6, 2), (12, 9), (13, 9), (14, 9), (15, 9),
    (17, 19), (18, 19), (77, 19), (20, 20),
)

CORE_LEAF_SEEDS = {
    0: ("what", "what?", "what!", "what is this", "what is this?", "what is that", "what do you want", "what do you do", "why", "why?", "why!", "why not", "why not?", "why me", "who", "who?", "who!", "who are you", "who are you?", "who is this", "who r u", "who is that", "hyw", "hyk", "iphey", "xhey", "hix", "hellod", "m bi", "u d?", "a bf here", "a gf here", "a boyfriend here", "a girlfriend here", "asdfgh", "qwerty", "salam", "bonjour", "salut", "good job", "good luck", "morning coffee", "help", "helicopter", "history", "supper", "super", "chicken", "pizza", "burger", "apple", "banana", "table", "chair", "window", "laptop"),
    1: ("hi", "hii", "hiii", "hiiii", "hi hi", "hey", "heyy", "heyyy", "heeyy", "hey hey", "hello", "helloo", "helo", "hlo", "ello", "yo", "yoo", "sup", "supp", "hiya", "heya", "howdy", "hy", "hyy", "hie", "hei", "hullo", "wassup", "wazzup", "whats up", "what's up", "what is up", "good morning", "morning", "mornin", "good afternoon", "good evening", "good day", "greetings", "welcome", "aloha", "hi there", "hey there", "hello there", "hey man", "hi man", "yo man", "hey cutie", "hi gorgeous", "hey handsome", "hey u", "hi u", "nice to meet u", "nice to meet you", "pleased to meet you", "good to see you", "long time no see"),
    2: ("age?", "ur age?", "your age?", "what is your age", "whats ur age", "how old are you", "how old r u", "how old are u", "how old u", "how old", "u how old?"),
    3: ("24", "22", "18", "30", "45", "im 24", "i'm 24", "lm 24", "l'm 22", "i am 25", "24y", "24 yo", "21yres", "20yres", "23 years old", "im24", "lm22"),
    4: ("from?", "u from?", "where from?", "where are you from", "where r u from", "where are u from", "where u from", "country?", "ur country?", "which country", "where do u live"),
    5: ("tunisia", "tuinsi", "tunis", "tun", "india", "handi", "ind", "usa", "uk", "france", "germany", "canada", "from usa", "im from usa", "lm from tunisia", "hi from tunis", "in uk"),
    6: ("hi l'm 30 from tun", "hi lm 24 from usa", "24 from uk", "im 22 from india", "25 tunisia", "im 28 in canada", "21 usa", "30 from france"),
    7: ("15", "16", "17", "14", "12", "im 15", "im 16", "i am 17", "15f", "16m", "14 girl", "17 boy"),
    8: ("how are you", "how are you?", "how are you!", "hru", "hru?", "hru!", "hyd", "hyd?", "how r u", "how are u", "how u doing", "how u doing!", "how are you doing", "how r u doing", "how's it going", "hows it going", "how was your day", "how is your day", "how have you been", "hor", "how ar you dopin", "how u doin", "hw r u", "how whas you da", "hi how r u", "hey hru", "hello how are you doing!", "hi how are you", "hey how r u"),
    9: ("m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m", "m/f?", "f/m?", "m f", "f m", "girl or boy", "boy or girl", "male or female", "female or male", "are you a boy or girl", "are u male or female", "what gender are you", "whats your gender", "ur gender", "gender?", "gender", "your sex", "what sex are you"),
    10: ("f?", "girl?", "female?", "woman?", "w?", "famle?", "u f?", "you f?", "are you a girl", "are u a girl", "r u a girl", "r u female", "are u female", "u a girl?", "is your gender female", "are you a woman"),
    11: ("m?", "boy?", "male?", "man?", "u m?", "you m?", "are you a boy", "are u a boy", "r u a boy", "r u male", "are u male", "u a boy?", "is your gender male", "are you a man", "are u a guy"),
    12: ("f", "female", "famle", "famel", "girl", "gril", "gurl", "woman", "w", "f here", "female here", "girl here", "im a girl", "i'm a girl", "lm a girl", "im female", "im famle", "hi f", "hey f", "lm f u", "im f u", "im f u?", "f u?", "im a girl u?", "girl here u?"),
    13: ("m", "male", "maale", "boy", "boi", "man", "guy", "m here", "male here", "boy here", "im a boy", "i'm a boy", "lm a boy", "im male", "im a guy", "hi m", "hey m", "lm m u", "im m u", "im m u?", "m u?", "im a boy u?", "m here u?"),
    14: ("22f", "22 f", "19 famle", "21 girl", "24 female", "f 22", "im 21 f", "im 23 girl", "20 gril", "25 woman"),
    15: ("24m", "24 m", "22 male", "21 boy", "25 guy", "m 24", "im 23 m", "im 26 boy", "20 man", "27 male"),
    16: ("wyd", "wyd?", "wud", "wud?", "what are you doing", "what are you doing?", "what r u doing", "what are u doing", "what u doing", "what ya doing", "what u doin", "whatcha doing", "what are you up to", "what r u up to", "what u up to", "whatcha up to", "doing anything", "doing anything fun", "wat u doing", "wht u doing", "wyd rn", "hi wyd", "hey what u doing"),
    17: ("im good", "i'm good", "lm good", "good", "doing good", "im doing good", "im great", "doing great", "im fine", "fine", "all good", "pretty good", "im ok", "im okay", "not bad", "im chat", "im chatting", "chatting", "just chatting", "chilling", "just chilling", "im chilling", "chilling in bed", "just chilling in bed", "relaxing", "in bed", "at home", "working", "just working", "studying", "listening to music", "watching tv", "watching a movie", "gaming", "playing games", "scrolling"),
    18: ("bored", "borad", "boerd", "borde", "im bored", "i'm bored", "lm bored", "lm borad", "im borad", "just bored", "so bored", "very bored", "bored in bed", "nothing", "nothin", "nothing much", "not much", "nm", "doing nothing", "nothing much just bored", "tired", "im tired", "lm tired", "sleepy", "im sleepy", "cant sleep", "sad", "im sad", "lonely", "im lonely"),
    19: ("im good u", "im good u?", "lm good u?", "good u?", "doing good u?", "im good wbu?", "im good hbu?", "im fine u?", "chilling u?", "just chilling u?", "bored u?", "im bored u?", "lm borad u?", "nothing much u?", "nothing much wbu?", "not much u?", "nm u?", "nmu", "nmu?", "tired u?", "chatting with u wbu?", "chatting with u, u?"),
    20: ("ur cute", "u r cute", "you are cute", "you're cute", "ur so cute", "u seem nice", "you seem nice", "u seem cool", "u seem sweet", "i like u", "i like you", "i really like u", "like u", "i like talking to u", "i like ur vibe", "love ur vibe", "u have a nice vibe", "ur hot", "ur pretty", "ur gorgeous", "ur handsome", "ur sweet", "u look good", "you look good", "u look cute", "ur funny", "ur cool", "ur awesome", "ur my type", "u make me smile", "hey ur cute", "aw ur sweet"),
    21: ("haha", "hahaha", "hehe", "lol", "lmao", "rofl", "nice", "cool", "sweet", "awesome", "amazing", "dope", "lit", "fire", "bet", "word", "fair", "fair enough", "oh nice", "oh cool", "haha nice", "lol cool", "sounds good", "sounds nice", "thats nice", "thats cool", "same", "samee", "sam", "samm", "same here", "sam here", "haha same", "lol same", "same lol", "sam lol", "same to u", "me too", "true", "tru", "so true", "thats true", "facts", "fr", "for real", "real", "ikr", "right", "agreed", "ok", "okay", "kk", "alright", "ok nice", "ok cool", "haha ok", "yeah", "yea", "yep", "yup", "sure", "ofc", "wow", "omg"),
    76: ("u", "u?", "you", "you?", "and u", "and u?", "and you", "and you?", "what about u", "what about u?", "what about you", "what about you?", "how about u", "how about u?", "how about you", "how about you?", "wbu", "wbu?", "hbu", "hbu?", "ur turn", "your turn", "and urself", "and yourself"),
    77: ("chatting with u", "chatting with you", "just chatting with u", "just chatting with you", "im chatting with u", "i'm chatting with u", "lm chatting with u", "im chatting with you", "chating with u", "chatin with u", "chatting with u rn", "chatting to u", "talking with u", "talking with you", "just talking with u", "im talking with u", "talking to u", "talking to you", "just talking to u", "im talking to u", "texting u", "texting you", "just texting u", "im texting u", "texting with u", "texting with you", "speaking with u", "speaking to u"),
}

ZERO_FAULT_SUITE = (
    (b"hi", 1, 1), (b"hi hi", 1, 1), (b"hy", 1, 1), (b"hyy", 1, 1), (b"hiiii", 1, 1), (b"hey", 1, 1), (b"hey hey", 1, 1), (b"heeyy", 1, 1), (b"hello", 1, 1), (b"helloo", 1, 1), (b"hlo", 1, 1), (b"yo", 1, 1), (b"sup", 1, 1), (b"good morning", 1, 1), (b"whats up", 1, 1), (b"hi there", 1, 1), (b"hey man", 1, 1), (b"nice to meet u", 1, 1), (b"hey cutie", 1, 1),
    (b"age?", 2, 2), (b"how old r u", 2, 2), (b"how old are you", 2, 2), (b"24", 3, 3), (b"im 24", 3, 3), (b"lm 24", 3, 3), (b"24y", 3, 3), (b"21yres", 3, 3), (b"im24", 3, 3), (b"from?", 4, 4), (b"where r u from", 4, 4), (b"where are you from", 4, 4), (b"tuinsi", 5, 5), (b"handi", 5, 5), (b"ind", 5, 5), (b"tunisia", 5, 5), (b"im from usa", 5, 5), (b"hi l'm 30 from tun", 6, 6), (b"24 from uk", 6, 6), (b"15", 7, 7), (b"im 16", 7, 7), (b"next", 99, 75), (b"/start", 99, 75),
    (b"how are you", 8, 8), (b"how are you?", 8, 8), (b"how are you!", 8, 8), (b"hru", 8, 8), (b"hyd", 8, 8), (b"how r u", 8, 8), (b"how u doing!", 8, 8), (b"how was your day", 8, 8), (b"hor", 8, 8), (b"how ar you dopin", 8, 8), (b"hw r u", 8, 8), (b"hi how r u", 8, 8), (b"hello how are you doing!", 8, 8),
    (b"m or f", 9, 9), (b"m or f?", 9, 9), (b"m/f", 9, 9), (b"f/m", 9, 9), (b"gender?", 9, 9), (b"what gender are you", 9, 9), (b"f?", 10, 10), (b"girl?", 10, 10), (b"are you a girl", 10, 10), (b"r u female", 10, 10), (b"u f?", 10, 10), (b"m?", 11, 11), (b"boy?", 11, 11), (b"are you a boy", 11, 11), (b"r u male", 11, 11), (b"u m?", 11, 11), (b"f", 12, 12), (b"famle", 12, 12), (b"f here", 12, 12), (b"im a girl", 12, 12), (b"hi f", 12, 12), (b"lm f u", 12, 12), (b"im f u?", 12, 12), (b"m", 13, 13), (b"male", 13, 13), (b"m here", 13, 13), (b"im a boy", 13, 13), (b"hi m", 13, 13), (b"im m u?", 13, 13), (b"22f", 14, 14), (b"24 m", 14, 15), (b"19 famle", 14, 14), (b"21 girl", 14, 14),
    (b"wyd", 15, 16), (b"wyd?", 15, 16), (b"what are you doing", 15, 16), (b"what r u doing", 15, 16), (b"what u up to", 15, 16),
    (b"im good", 16, 17), (b"im ok", 16, 17), (b"just chilling in bed", 16, 17), (b"just chatting", 16, 17), (b"listening to music", 16, 17), (b"Chatting with u", 16, 77), (b"chatting with you", 16, 77), (b"talking with u", 16, 77), (b"texting u", 16, 77), (b"im chatting with u", 16, 77), (b"bored", 17, 18), (b"lm borad", 17, 18), (b"nothing much", 17, 18), (b"im tired", 17, 18), (b"im good u?", 18, 19), (b"lm borad u?", 18, 19), (b"nmu", 18, 19), (b"nothing much wbu?", 18, 19),
    (b"ur cute", 19, 20), (b"u seem nice", 19, 20), (b"i like u", 19, 20), (b"u look good", 19, 20), (b"love ur vibe", 19, 20), (b"haha", 20, 21), (b"lol", 20, 21), (b"nice", 20, 21), (b"cool", 20, 21), (b"true", 20, 21), (b"ok", 20, 21), (b"same", 20, 21), (b"sam", 20, 21), (b"same here", 20, 21), (b"sam here", 20, 21), (b"for real", 20, 21),
    (b"u", 0, 76), (b"u?", 0, 76), (b"and u?", 0, 76), (b"wbu?", 0, 76), (b"what about u?", 0, 76),
    (b"Chicken", 0, 0), (b"what", 0, 0), (b"what?", 0, 0), (b"what!", 0, 0), (b"what do you do", 0, 0), (b"what is this", 0, 0), (b"why", 0, 0), (b"why?", 0, 0), (b"why not", 0, 0), (b"who", 0, 0), (b"who?", 0, 0), (b"who are you", 0, 0), (b"hyw", 0, 0), (b"iphey", 0, 0), (b"m bi", 0, 0), (b"a bf here", 0, 0), (b"a gf here", 0, 0), (b"asdfgh", 0, 0),
)


def load_txt_samples(sample_map, load_fn):
    direct = (
        ("greetings.txt", 1, 800), ("hru.txt", 8, 800), ("gender_ask_choice.txt", 9, 400),
        ("gender_ask_is_female.txt", 10, 400), ("gender_ask_is_male.txt", 11, 400),
        ("gender_state_female.txt", 12, 400), ("gender_state_male.txt", 13, 400),
        ("wyd.txt", 16, 600), ("status_positive.txt", 17, 700), ("status_negative.txt", 18, 700),
        ("status_ask_back.txt", 19, 700), ("compliments_flirt.txt", 20, 700), ("reactions_ack.txt", 21, 700),
    )
    for fn, lf, cap in direct:
        for s in load_fn(fn, cap):
            if fn == "hru.txt" and s.rstrip("?") in ("wbu", "hbu"):
                continue
            sample_map[s] = (lf, np.float32(4.0))

    ask_age_re = re.compile(r"\b(?:how\s+old|ur\s+age|your\s+age)\b|^age\??$", re.I)
    ask_from_re = re.compile(r"\b(?:where\s+(?:are\s+you|r\s+u|u)\s+from|where\s+do\s+(?:you|u)\s+live|which\s+country|ur\s+country|u\s+from)\b|^from\??$|^country\??$", re.I)
    adult_re = re.compile(r"\b(1[89]|[2-7][0-9]|80)\b")
    for s in load_fn("ages.txt", 600):
        sample_map[s] = (2, np.float32(10.0)) if ask_age_re.search(s) else (3, np.float32(4.0))
    for s in load_fn("countries.txt", 800):
        sample_map[s] = (4, np.float32(10.0)) if ask_from_re.search(s) else ((6, np.float32(6.0)) if adult_re.search(s) else (5, np.float32(4.0)))
    for s in load_fn("age_gender.txt", 500):
        sample_map[s] = (14 if any(tok in s for tok in ("f", "girl", "gril", "woman")) else 15, np.float32(5.0))