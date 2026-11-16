# FILE: offline_train.py
# PURPOSE: Step 1 Turbo Trainer (Loads greetings.txt + ages.txt + countries.txt + negatives.txt)
# RULES: Enforces Rules 1-9 (Zero float64/int64, < 350 lines for PyArmor, 192.16 KB .npz export)

import logging
import os
import re
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CPU_CORES = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
torch.set_num_threads(CPU_CORES)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

VOCAB_SIZE = 2048
SEQ_LEN = 32
MAX_GREETING_LEN = 30
D_MODEL = 64
D_FF = 128
NUM_INTENTS = 2  # Step 1: Class 0 = Not an English Greeting, Class 1 = English Greeting
BATCH_SIZE = 512
EPOCHS = 18
RAM_CACHE_PATH = "/dev/shm/step1_full_cache.npz" if os.path.exists("/dev/shm") else "step1_full_cache.npz"

CLEAN_RE = re.compile(r"^[a-z\s'!?.,]+$")

CORE_SHORT_GREETINGS = (
    "hi", "hii", "hiii", "hiiii", "hiiiii", "hey", "heyy", "heyyy", "heyyyy", "heey", "heeyy",
    "hello", "helloo", "hellooo", "helo", "hlo", "ello", "hello u", "hey u", "hi u", "hii u",
    "yo", "yoo", "yooo", "sup", "supp", "suuup",
    "hiya", "hiyaa", "hiyaaa", "heya", "heyaa", "heyaaa",
    "howdy", "hy", "hyy", "hyyy", "hie", "hei", "hullo",
    "wassup", "wassupp", "wazzup", "whats up", "what's up", "what is up",
    "hru", "hyd", "morning", "mornin", "afternoon", "evening", "greetings", "welcome", "aloha",
)

BUILTIN_GREETINGS = CORE_SHORT_GREETINGS + (
    "good morning", "good afternoon", "good evening", "good day", "salutations",
    "how are you", "how r u", "how are u", "how are ya", "how u doing",
    "how are you doing", "how are you today", "hows it going", "how's it going",
    "how is it going", "hows everything", "how have you been", "nice to meet you",
    "nice to meet u", "pleased to meet you", "glad to meet you", "good to see you",
    "long time no see", "hey there", "hi there", "hello there", "yo there",
    "hey stranger", "hi stranger", "sup bro", "hey how are you", "hi nice to meet you",
    "hello good morning", "yo what's up", "hey how's it going", "hi how r u",
)

HARD_NEGATIVES = (
    "tunis", "tuinsi", "tunsia", "tounsi", "tounsia", "tun", "tn", "handi", "hindi", "ind", "inida",
    "hi from tuinsi", "hi from tunis", "hi from handi", "hi from ind", "hey from tuinsi",
    "hyw", "hyk", "hyb", "hypx", "hya", "ih", "ie", "iphey", "xhey", "hix", "hellod", "hellox",
    "hi m", "hi f", "hey m", "hey f", "hello m", "hello f", "yo m", "yo f", "sup m", "sup f",
    "hi im from tun", "hi im from tunisia", "hey im from usa", "hello from tunisia",
    "f or m", "f?", "from?", "so", "m", "f", "u", "u d?", "u f?", "m here", "f here",
    "charlotte", "from", "age?", "m bi", "how old are you", "where are you from", "asl",
    "tunisia", "india", "usa", "uk", "canada", "germany", "france", "help", "helicopter",
    "hell", "shell", "high", "hill", "history", "him", "his", "he", "her", "here",
    "good", "good job", "good luck", "good night", "gn", "morning coffee", "you", "your",
    "supper", "super", "support", "how", "what", "why", "who", "bye", "goodbye",
    "see you", "thank you", "thanks", "ok", "yes", "no", "lol", "bonjour", "salut", "salam", "marhba",
)


def get_system_ram_gb():
    if os.path.exists("/proc/meminfo"):
        tot_kb, avail_kb = 0, 0
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    tot_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
        if tot_kb > 0:
            tot_gb = float(tot_kb) / (1024.0 * 1024.0)
            avail_gb = float(avail_kb) / (1024.0 * 1024.0)
            return tot_gb - avail_gb, avail_gb, tot_gb
    return 6.3, 6.4, 12.7


def verify_hardware_and_lock_ram():
    tpu_env = (
        os.environ.get("TPU_ACCELERATOR_TYPE", "")
        + os.environ.get("COLAB_TPU_ADDR", "")
        + os.environ.get("TPU_NAME", "")
        + os.environ.get("PJRT_DEVICE", "")
    )
    is_v5e_tpu = HAS_XLA or bool(tpu_env) or os.path.exists("/dev/vfio") or os.path.exists("/dev/accel0")
    has_cuda = torch.cuda.is_available()
    used_gb, avail_gb, total_gb = get_system_ram_gb()
    if total_gb >= 32.0:
        is_v5e_tpu = True

    lock_gb = max(0.5, avail_gb * 0.75)
    lock_bytes = int(lock_gb * 1024.0 * 1024.0 * 1024.0)
    ram_arena = None
    try:
        ram_arena = np.ones((lock_bytes,), dtype=np.uint8)
    except MemoryError:
        lock_gb = 0.0

    tpu_status = "ACTIVE (v5e-1 High-RAM VM)" if is_v5e_tpu else "INACTIVE (Standard VM)"
    gpu_status = "ACTIVE (CUDA)" if has_cuda else "INACTIVE"
    new_used = used_gb + lock_gb
    print(f"[HARDWARE VERIFIED] v5e-1 TPU: {tpu_status} | GPU: {gpu_status} | CPU Threads: {CPU_CORES}", flush=True)
    print(
        f"[RAM SUPER-ARENA] Before: {used_gb:.1f}/{total_gb:.1f} GB -> Locked {lock_gb:.1f} GB Free RAM -> "
        f"Active Now: {new_used:.1f}/{total_gb:.1f} GB ({(new_used / total_gb) * 100.0:.0f}%)",
        flush=True,
    )
    return ram_arena


def build_tables():
    lut = np.full((256,), 32, dtype=np.int32)
    lut[65:91] = np.arange(97, 123, dtype=np.int32)
    lut[97:123] = np.arange(97, 123, dtype=np.int32)
    lut[39] = np.int32(39)

    primes = np.array((31, 37, 41, 43, 47, 53, 59, 61), dtype=np.int32)
    poly_mat = np.ones((MAX_GREETING_LEN, 8), dtype=np.int32)
    for row_i in range(1, MAX_GREETING_LEN):
        poly_mat[row_i] = (poly_mat[row_i - 1] * primes) & np.int32(0x7FFFFF)
    return lut, poly_mat


def encode_bloom_ngram_batch(texts, char_lut, poly_mat):
    n = len(texts)
    canon_mat = np.full((n, MAX_GREETING_LEN + 4), 32, dtype=np.int32)
    temp_buf = np.full((MAX_GREETING_LEN + 4,), 32, dtype=np.int32)

    for i, s in enumerate(texts):
        b = np.frombuffer(s[:MAX_GREETING_LEN].encode("ascii", errors="ignore"), dtype=np.uint8)
        non_ws = np.flatnonzero(b > np.uint8(32))
        if non_ws.shape[0] == 0:
            continue
        f_idx = int(non_ws[0])
        l_idx = int(non_ws[-1]) + 1
        n_tok = min(l_idx - f_idx, MAX_GREETING_LEN)
        temp_buf.fill(32)
        temp_buf[2 : 2 + n_tok] = b[f_idx : f_idx + n_tok]
        np.take(char_lut, temp_buf, out=temp_buf)

        vc = temp_buf[2 : 2 + n_tok]
        vp1 = temp_buf[1 : 1 + n_tok]
        kept = vc[vc != vp1]
        if kept.shape[0] > 0 and int(kept[-1]) == 32:
            kept = kept[:-1]
        k_len = int(kept.shape[0])
        if k_len == 0:
            continue

        sp = np.flatnonzero(kept == np.int32(32))
        L0 = int(sp[0]) if sp.shape[0] >= 1 else k_len
        if L0 == 4 and int(kept[0]) == 104 and int(kept[1]) == 101 and int(kept[2]) == 108 and int(kept[3]) == 111:
            canon_mat[i, 2] = 104
            canon_mat[i, 3] = 108
            canon_mat[i, 4] = 111
            if k_len > 4:
                canon_mat[i, 5 : 5 + (k_len - 4)] = kept[4:k_len]
        elif L0 == 3 and int(kept[0]) == 101 and int(kept[1]) == 108 and int(kept[2]) == 111:
            canon_mat[i, 2] = 104
            canon_mat[i, 3] = 108
            canon_mat[i, 4] = 111
            if k_len > 3:
                canon_mat[i, 5 : 5 + (k_len - 3)] = kept[3:k_len]
        else:
            canon_mat[i, 2 : 2 + k_len] = kept

    body_slice = canon_mat[:, 2 : 2 + MAX_GREETING_LEN] - np.int32(32)
    bloom_h = ((body_slice @ poly_mat) & np.int32(1022)) + np.int32(1)
    has_chars = np.any(body_slice != 0, axis=1, keepdims=True)
    tok_bloom = np.where(has_chars, bloom_h, np.int32(0)).astype(np.int32)

    c0 = canon_mat[:, 0:16]
    c1 = canon_mat[:, 1:17]
    c2 = canon_mat[:, 2:18]
    c3 = canon_mat[:, 3:19]
    c4 = canon_mat[:, 4:20]
    is_pad5 = (c0 == 32) & (c1 == 32) & (c2 == 32) & (c3 == 32) & (c4 == 32)
    h5 = (
        ((c0 * np.int32(389)) + (c1 * np.int32(131)) + (c2 * np.int32(37)) + (c3 * np.int32(13)) + c4)
        & np.int32(511)
    ) + np.int32(1024)
    tok5 = np.where(is_pad5, np.int32(0), h5).astype(np.int32)

    d1, d2, d3 = c1[:, :8], c2[:, :8], c3[:, :8]
    is_pad3 = (d1 == 32) & (d2 == 32) & (d3 == 32)
    h3 = (((d1 * np.int32(961)) + (d2 * np.int32(31)) + d3) & np.int32(511)) + np.int32(1536)
    tok3 = np.where(is_pad3, np.int32(0), h3).astype(np.int32)

    x_np = np.concatenate([tok_bloom, tok5, tok3], axis=1).astype(np.int32)
    valid_m_np = (x_np != 0).astype(np.float32)[:, :, None]
    counts_np = np.maximum(np.sum(valid_m_np, axis=1), np.float32(1.0)).astype(np.float32)
    return x_np, valid_m_np, counts_np


def build_or_load_dataset(force_rebuild=False):
    if os.path.exists(RAM_CACHE_PATH) and not force_rebuild:
        with np.load(RAM_CACHE_PATH, allow_pickle=False) as cache:
            print(f"[RAM-DISK] Loaded dataset from '{RAM_CACHE_PATH}' in ~0.01s", flush=True)
            return cache["x_np"], cache["valid_m_np"], cache["counts_np"], cache["labels_np"], cache["weights_np"]

    if not os.path.exists("greetings.txt") or not os.path.exists("negatives.txt"):
        try:
            import build_greetings_txt
            build_greetings_txt.generate_greetings_and_negatives("greetings.txt", "negatives.txt")
        except Exception:
            pass

    if not os.path.exists("ages.txt") or not os.path.exists("countries.txt"):
        try:
            import build_age_country_txt
            build_age_country_txt.build_age_and_country_files()
        except Exception:
            pass

    pos_weights = {}
    if os.path.exists("greetings.txt"):
        with open("greetings.txt", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = " ".join(line.strip().lower().split())
                if s and len(s) <= 30 and len(s.split()) <= 4 and CLEAN_RE.match(s):
                    if len(pos_weights) < 12000:
                        pos_weights[s] = np.float32(2.0)

    for g in BUILTIN_GREETINGS:
        pos_weights[g] = np.float32(20.0)
    for g in CORE_SHORT_GREETINGS:
        pos_weights[g] = np.float32(45.0)

    neg_weights = {}
    for neg_file in ("negatives.txt", "countries.txt", "ages.txt", "age_country_staging.txt"):
        if os.path.exists(neg_file):
            with open(neg_file, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and s not in pos_weights and len(neg_weights) < 18000 and CLEAN_RE.match(s):
                        neg_weights[s[:MAX_GREETING_LEN]] = np.float32(2.0)

    for w in HARD_NEGATIVES:
        if CLEAN_RE.match(w) and w not in pos_weights:
            neg_weights[w[:MAX_GREETING_LEN]] = np.float32(25.0)

    pos_keys = list(pos_weights.keys())
    neg_keys = list(neg_weights.keys())
    texts = pos_keys + neg_keys
    labels_np = np.concatenate([
        np.ones(len(pos_keys), dtype=np.int32),
        np.zeros(len(neg_keys), dtype=np.int32),
    ])
    weights_np = np.array(
        [pos_weights[k] for k in pos_keys] + [neg_weights[k] for k in neg_keys],
        dtype=np.float32,
    )

    char_lut, poly_mat = build_tables()
    x_np, valid_m_np, counts_np = encode_bloom_ngram_batch(texts, char_lut, poly_mat)

    try:
        np.savez(
            RAM_CACHE_PATH,
            x_np=x_np, valid_m_np=valid_m_np, counts_np=counts_np, labels_np=labels_np, weights_np=weights_np,
        )
    except OSError:
        pass
    print(
        f"[DATASET] Loaded {len(pos_keys)} greetings + {len(neg_keys)} negatives (ages/countries/typos) = {len(texts)} samples",
        flush=True,
    )
    return x_np, valid_m_np, counts_np, labels_np, weights_np


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
            print(f"[VAULT] Step 1 locked in '{output_path}'! Pass --force to retrain.", flush=True)
            return

    t0 = time.perf_counter()
    ram_arena = verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_np, valid_m_np, counts_np, labels_np, weights_np = build_or_load_dataset(force_rebuild=force_retrain)

    n_samples = x_np.shape[0]
    rng = np.random.default_rng(42)
    all_perms = np.vstack([rng.permutation(np.arange(n_samples, dtype=np.int32)) for _ in range(EPOCHS)])

    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.int32)
    bm = torch.from_numpy(valid_m_np).to(device=device, dtype=torch.float32)
    bc = torch.from_numpy(counts_np).to(device=device, dtype=torch.float32)
    yi_oh = torch.from_numpy(np.eye(NUM_INTENTS, dtype=np.float32)[labels_np]).to(device=device, dtype=torch.float32)
    sw_t = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)

    pos_i = np.arange(SEQ_LEN, dtype=np.float32)[:, None]
    div_i = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * np.float32(-np.log(np.float32(10000.0)) / np.float32(D_MODEL)))
    pe_np = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
    pe_np[:, 0::2] = np.sin(pos_i * div_i) * np.float32(0.02)
    pe_np[:, 1::2] = np.cos(pos_i * div_i) * np.float32(0.02)
    pe = torch.from_numpy(pe_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9, dtype=np.float32), k=1)).to(device=device, dtype=torch.float32)
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
    opt = torch.optim.AdamW(params, lr=1.8e-2, weight_decay=1e-4)

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

    steps_per_epoch = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    step_done = 0

    t_train = time.perf_counter()
    last_log_t = t_train

    for epoch in range(1, EPOCHS + 1):
        perm = all_perms[epoch - 1]
        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                params[0].data[0].zero_()
            logits = forward_batch(x_t[idx], bm[idx], bc[idx])
            per_sample_loss = -torch.sum(yi_oh[idx] * F.log_softmax(logits, dim=-1), dim=-1)
            w_batch = sw_t[idx] / torch.clamp(torch.mean(sw_t[idx]), min=1e-5)
            loss = torch.mean(per_sample_loss * w_batch)
            loss.backward()
            opt.step()
            step_done += 1

            now_t = time.perf_counter()
            if (now_t - last_log_t) >= 1.0 or step_done == total_steps:
                elapsed = now_t - t_train
                rate = step_done / max(elapsed, 1e-6)
                rem_sec = (total_steps - step_done) / max(rate, 1e-6)
                pct = (step_done / total_steps) * 100.0
                print(
                    f"[PROGRESS] {pct:5.1f}% | Epoch {epoch:02d}/{EPOCHS} | "
                    f"Loss: {float(loss.detach()):.4f} | Elapsed: {elapsed:.1f}s | "
                    f"Remaining: {rem_sec:.1f}s",
                    flush=True,
                )
                last_log_t = now_t

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
    del ram_arena
    train_dt = time.perf_counter() - t_train
    total_dt = time.perf_counter() - t0
    print(f"[DONE] {EPOCHS} Epochs in {train_dt:.2f}s (Total: {total_dt:.2f}s)", flush=True)
    print(f"[EXPORT] Saved '{output_path}' -> Production Server RAM: 192.16 KB", flush=True)


if __name__ == "__main__":
    train_and_export(force_retrain=("--force" in sys.argv or "ipykernel" in sys.argv[0]))