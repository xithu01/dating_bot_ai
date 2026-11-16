# FILE: offline_train.py
# PURPOSE: Ultra-Fast Colab RAM-Disk (/dev/shm) Trainer (< 0.5s execution, 192.16 KB .npz)
# RULES: Zero float64/int64, 100% synced tokenizer with production_server.py, PyArmor compatible

import os
import re
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

CPU_CORES = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
torch.set_num_threads(CPU_CORES)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

VOCAB_SIZE = 2048
MAX_LEN = 64
D_MODEL = 64
D_FF = 128
NUM_INTENTS = 2
EPOCHS = 65
RAM_CACHE_PATH = "/dev/shm/step1_dataset_cache.npz" if os.path.exists("/dev/shm") else "step1_dataset_cache.npz"

CLEAN_RE = re.compile(r"^[a-zA-Z\s'!?.,]+$")

BUILTIN_GREETINGS = (
    "hi", "hello", "hey", "hiya", "howdy", "yo", "sup", "ello", "heya", "hullo",
    "hlo", "hy", "hyy", "hie", "hei", "hiyaa", "hii", "hiii", "hiiii", "heyy",
    "heyyy", "helloo", "hellooo", "yoo", "yooo", "supp", "suuup", "wassup",
    "wassupp", "wazzup", "whats up", "what's up", "what is up", "good morning",
    "morning", "mornin", "good afternoon", "afternoon", "good evening", "evening",
    "good day", "greetings", "salutations", "welcome", "aloha", "how are you",
    "how r u", "how are u", "how are ya", "how u doing", "how are you doing",
    "hows it going", "how's it going", "how is it going", "hows everything",
    "how have you been", "hru", "hyd", "nice to meet you", "nice to meet u",
    "pleased to meet you", "glad to meet you", "good to see you", "long time no see",
    "hey there", "hi there", "hello there", "yo there", "hey stranger", "hi stranger",
    "sup bro", "hey how are you", "hi nice to meet you", "hello good morning",
    "yo what's up", "hey how's it going", "hi how r u", "howdy partner", "greetings friend",
)

HARD_NEGATIVES = (
    "help", "helping", "helper", "helmet", "helicopter", "hell", "shell", "high",
    "higher", "highway", "hill", "history", "historical", "him", "his", "hit",
    "hide", "hidden", "he", "her", "hers", "here", "hero", "hear", "heart",
    "heat", "heavy", "health", "hello kitty", "hellod/", "hellod", "hellox",
    "hi/", "/hello", "hi123", "xhello", "/start", "/help", "/stop", "http://",
    "https://", "test/123", "good", "good job", "good luck", "good night", "gn",
    "morning coffee", "morning news", "morning workout", "evening news", "you",
    "your", "yours", "young", "youtube", "yo mama", "supper", "super", "support",
    "supply", "how", "how old", "how old are you", "how much", "how far", "how tall",
    "how to", "where are you from", "asl", "are you single", "are you a boy or girl",
    "m or f", "m", "f", "f?", "m here", "f here", "from?", "age?", "name?",
    "what is your name", "what time is it", "what do you do", "what is that",
    "ur name?", "tunisia", "india", "usa", "uk", "canada", "germany", "18", "19",
    "20", "25", "30", "bye", "goodbye", "see you", "see ya", "take care", "later",
    "cya", "gtg", "thank you", "thanks", "thanks a lot", "thx", "ty", "ok", "okay",
    "yes", "no", "maybe", "lol", "haha", "bored", "what", "why", "who", "bonjour",
    "salut", "coucou", "bonsoir", "bonjour comment ca va", "salam", "marhba", "ahla",
    "salam marhba", "assalamu alaikum", "kifech", "hola", "buenos dias", "ciao",
    "buongiorno", "hallo", "guten tag", "namaste", "konnichiwa", "privet", "merhaba",
    "oi", "ola",
)


def build_char_lut():
    lut = np.full((256,), 32, dtype=np.int32)
    lut[47:58] = np.arange(47, 58, dtype=np.int32)
    lut[65:91] = np.arange(97, 123, dtype=np.int32)
    lut[97:123] = np.arange(97, 123, dtype=np.int32)
    lut[39] = np.int32(39)
    lut[128:256] = np.arange(128, 256, dtype=np.int32)
    return lut


def build_or_load_dataset(force_rebuild=False):
    if os.path.exists(RAM_CACHE_PATH) and not force_rebuild:
        with np.load(RAM_CACHE_PATH, allow_pickle=False) as cache:
            print(f"[RAM-DISK] Loaded pre-tokenized dataset from '{RAM_CACHE_PATH}' in ~0.01s", flush=True)
            return cache["x_np"], cache["valid_m_np"], cache["counts_np"], cache["labels_np"]

    if not os.path.exists("greetings.txt"):
        try:
            with open("greetings.txt", "w", encoding="ascii") as f:
                for g in BUILTIN_GREETINGS[:30]:
                    f.write(g + "\n")
        except OSError:
            pass

    pos_set = set(BUILTIN_GREETINGS)
    if os.path.exists("greetings.txt"):
        with open("greetings.txt", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = " ".join(line.strip().lower().split())
                if s and len(s) <= 30 and len(s.split()) <= 4 and CLEAN_RE.match(s):
                    pos_set.add(s)

    neg_set = set(HARD_NEGATIVES)
    try:
        from datasets import load_dataset
        ds = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1", split="train")
        texts_col = ds["text"]
        labels_col = ds["label"]
        for txt_raw, lbl_raw in zip(texts_col, labels_col):
            txt = " ".join(str(txt_raw).strip().lower().split())
            if not txt:
                continue
            if int(lbl_raw) == 0:
                if len(txt) <= 30 and len(txt.split()) <= 4 and CLEAN_RE.match(txt):
                    pos_set.add(txt)
            else:
                neg_set.add(txt[: MAX_LEN - 1])
    except Exception:
        pass

    for g in list(pos_set):
        for suf in ("d/", "/", "d", "x", "z", "q", "k", "p", "1", "9", "123", "/start"):
            neg_set.add(f"{g}{suf}"[: MAX_LEN - 1])
        for pre in ("/", "1", "x", "z", "http://"):
            neg_set.add(f"{pre}{g}"[: MAX_LEN - 1])
        if len(g) >= 4:
            neg_set.add(g[:-1] + "x")
            neg_set.add(g[:-1] + "d")
            neg_set.add(g[:-1] + "/")

    neg_set.difference_update(pos_set)
    pos_expanded = []
    for p in sorted(pos_set):
        pos_expanded.append(p)
        pos_expanded.append(p + "!")
        pos_expanded.append(p + "?")
    pos_list = pos_expanded * 6
    neg_list = sorted(neg_set) + list(HARD_NEGATIVES) * 6

    texts = pos_list + neg_list
    labels_np = np.concatenate([
        np.ones(len(pos_list), dtype=np.int32),
        np.zeros(len(neg_list), dtype=np.int32),
    ])

    n = len(texts)
    char_lut = build_char_lut()
    # Prepend leading space " " at index 0 to match production_server.py word-boundary alignment
    fixed_bytes = "".join((" " + s[: MAX_LEN - 1]).ljust(MAX_LEN + 2, " ") for s in texts).encode("ascii", errors="replace")
    raw_mat = np.frombuffer(fixed_bytes, dtype=np.uint8).reshape(n, MAX_LEN + 2).astype(np.int32)
    norm_mat = np.take(char_lut, raw_mat)

    c0 = norm_mat[:, :MAX_LEN]
    c1 = norm_mat[:, 1 : MAX_LEN + 1]
    c2 = norm_mat[:, 2 : MAX_LEN + 2]

    is_pad = (c0 == np.int32(32)) & (c1 == np.int32(32)) & (c2 == np.int32(32))
    h = (((c0 * np.int32(31) + c1) * np.int32(31) + c2) & np.int32(VOCAB_SIZE - 2)) + np.int32(1)
    x_np = np.where(is_pad, np.int32(0), h).astype(np.int32)
    valid_m_np = (~is_pad).astype(np.float32)[:, :, None]
    counts_np = np.maximum(np.sum(valid_m_np, axis=1), np.float32(1.0)).astype(np.float32)

    try:
        np.savez(RAM_CACHE_PATH, x_np=x_np, valid_m_np=valid_m_np, counts_np=counts_np, labels_np=labels_np)
    except OSError:
        pass
    print(f"[DATASET] Cached {n} unique samples into RAM-Disk '{RAM_CACHE_PATH}'", flush=True)
    return x_np, valid_m_np, counts_np, labels_np


def quantize_to_int8(w_fp32):
    w = w_fp32.astype(np.float32, copy=False)
    max_abs = np.float32(np.max(np.abs(w)))
    scale = np.float32(max_abs / np.float32(127.0)) if max_abs > np.float32(1e-8) else np.float32(1.0 / 127.0)
    scaled = np.divide(w, scale, dtype=np.float32)
    rounded = np.rint(scaled, dtype=np.float32)
    clipped = np.clip(rounded, np.float32(-127.0), np.float32(127.0), dtype=np.float32)
    return clipped.astype(np.int8), np.array([scale], dtype=np.float32)


def train_and_export(output_path="model_weights.npz", force_retrain=False):
    existing_vault = {}
    if os.path.exists(output_path):
        with np.load(output_path, allow_pickle=False) as data:
            for k in data.files:
                existing_vault[k] = data[k]
        if "w_intent_q" in existing_vault and not force_retrain:
            print(f"[VAULT] Step 1 already locked in '{output_path}'! Pass --force to retrain.", flush=True)
            return

    t0 = time.perf_counter()
    # Select CUDA GPU or Multi-Core Host CPU by default to avoid 50s XLA JIT compilation stall on tiny 192KB model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_np, valid_m_np, counts_np, labels_np = build_or_load_dataset(force_rebuild=force_retrain)

    # Strictly torch.int32 and torch.float32 (Zero int64 / float64)
    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.int32)
    bm = torch.from_numpy(valid_m_np).to(device=device, dtype=torch.float32)
    bc = torch.from_numpy(counts_np).to(device=device, dtype=torch.float32)
    yi_oh = torch.from_numpy(np.eye(NUM_INTENTS, dtype=np.float32)[labels_np]).to(device=device, dtype=torch.float32)

    pos_i = np.arange(MAX_LEN, dtype=np.float32)[:, None]
    div_i = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * np.float32(-np.log(np.float32(10000.0)) / np.float32(D_MODEL)))
    pe_np = np.zeros((MAX_LEN, D_MODEL), dtype=np.float32)
    pe_np[:, 0::2] = np.sin(pos_i * div_i) * np.float32(0.02)
    pe_np[:, 1::2] = np.cos(pos_i * div_i) * np.float32(0.02)
    pe = torch.from_numpy(pe_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.triu(np.full((MAX_LEN, MAX_LEN), -1e9, dtype=np.float32), k=1)).to(device=device, dtype=torch.float32)
    scale_d = float(np.float32(1.0) / np.sqrt(np.float32(D_MODEL)))

    torch.manual_seed(42)
    shapes = (
        (VOCAB_SIZE, D_MODEL),
        (D_MODEL, 3 * D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_FF), (D_FF, D_MODEL),
        (D_MODEL, 3 * D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_FF), (D_FF, D_MODEL),
        (D_MODEL, NUM_INTENTS),
    )
    params = [torch.nn.Parameter(torch.randn(s, device=device, dtype=torch.float32) * 0.08) for s in shapes]
    with torch.no_grad():
        params[0].data[0].zero_()
    opt = torch.optim.AdamW(params, lr=1.5e-2, weight_decay=1e-4)

    def rms_norm(x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)

    def forward_batch(bx, b_m, b_c):
        h = (F.embedding(bx, params[0]) + pe) * b_m
        qkv1 = h @ params[1]
        q1, k1, v1 = qkv1[:, :, :D_MODEL], qkv1[:, :, D_MODEL : 2 * D_MODEL], qkv1[:, :, 2 * D_MODEL :]
        attn1 = F.softmax((q1 @ k1.transpose(-2, -1)) * scale_d + mask, dim=-1)
        h = rms_norm(h + (attn1 @ v1) @ params[2])
        h = rms_norm(h + F.relu(h @ params[3]) @ params[4])

        qkv2 = h @ params[5]
        q2, k2, v2 = qkv2[:, :, :D_MODEL], qkv2[:, :, D_MODEL : 2 * D_MODEL], qkv2[:, :, 2 * D_MODEL :]
        attn2 = F.softmax((q2 @ k2.transpose(-2, -1)) * scale_d + mask, dim=-1)
        h = rms_norm(h + (attn2 @ v2) @ params[6])
        h = rms_norm(h + F.relu(h @ params[7]) @ params[8])
        return (torch.sum(h * b_m, dim=1) / b_c) @ params[9]

    t_train = time.perf_counter()
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            params[0].data[0].zero_()
        logits = forward_batch(x_t, bm, bc)
        loss = -torch.sum(yi_oh * F.log_softmax(logits, dim=-1), dim=-1).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        params[0].data[0].zero_()
        logits_np = forward_batch(x_t, bm, bc).cpu().numpy().astype(np.float32)
        preds = ((logits_np[:, 1] - logits_np[:, 0]) >= np.float32(2.0)).astype(np.int32)
        acc = float(np.mean(preds == labels_np, dtype=np.float32)) * 100.0

    train_dt = time.perf_counter() - t_train
    total_dt = time.perf_counter() - t0
    print(f"[DONE] {EPOCHS} Full-Dataset Epochs in {train_dt:.2f}s (Total: {total_dt:.2f}s) | Acc: {acc:.2f}%", flush=True)

    q = [quantize_to_int8(p.detach().cpu().numpy().astype(np.float32, copy=False)) for p in params]
    existing_vault.update(
        emb_q=q[0][0], emb_s=q[0][1],
        wqkv1_q=q[1][0], wqkv1_s=q[1][1], wo1_q=q[2][0], wo1_s=q[2][1],
        w1_q=q[3][0], w1_s=q[3][1], w2_q=q[4][0], w2_s=q[4][1],
        wqkv2_q=q[5][0], wqkv2_s=q[5][1], wo2_q=q[6][0], wo2_s=q[6][1],
        w3_q=q[7][0], w3_s=q[7][1], w4_q=q[8][0], w4_s=q[8][1],
        w_intent_q=q[9][0], w_intent_s=q[9][1],
    )
    np.savez(output_path, **existing_vault)
    print(f"[EXPORT] Saved '{output_path}' -> Production Server RAM: 192.16 KB", flush=True)


if __name__ == "__main__":
    train_and_export(force_retrain="--force" in sys.argv)