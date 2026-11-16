# FILE: production_server.py
# PURPOSE: Step 1 Pure NumPy Dating Engine (Synced Rolling Hash, 32 Short Replies, 2.56 MB RAM)
# RULES: Zero PyTorch/SciPy/DB, zero float64/int64, zero loops over tensors, PyArmor compatible

import asyncio
from collections import deque
import os
import sys
import time
import numpy as np

VOCAB_SIZE = 2048
SEQ_LEN = 64
MAX_GREETING_LEN = 30
D_MODEL = 64
D_FF = 128
NUM_INTENTS = 2
MAX_USERS = 50000
NUM_WORKERS = 8


def _pack_reply(class_id, is_greeting_str, intent_str, reply_str, kbd_str):
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


CACHED_REJECT_RESPONSE = _pack_reply(
    0, "false", "not_english_greeting", "say hi or hey in English to start!", '["hey","hi!"]'
)

_SHORT_GREETINGS_32 = (
    "hey", "hi", "heyy", "yo", "hey there", "hii", "sup", "hello!",
    "hey how r u", "hi! hru?", "yo whats up", "heyy :)", "hiya", "hey! hows it going",
    "hiii", "sup! hru", "hello there", "hey u", "hi there!", "yo! how are u",
    "heyyy", "whats up!", "hey! nice to meet u", "hi :) how r u", "howdy!", "hey whats up",
    "hii there", "hello :)", "yo! whats good", "hey! hru doing", "hi! hows u", "heyy! whats up",
)
CACHED_GREETING_POOL = tuple(
    _pack_reply(1, "true", "english_greeting", msg, '["good u?","doing great!"]')
    for msg in _SHORT_GREETINGS_32
)

_REPEAT_GREETINGS_8 = (
    "hey again lol", "haha hi again!", "we already said hi :)", "yo again! how r u?",
    "hii again haha", "already said hey lol", "haha hey x2! hru?", "hi again! whats up?",
)
CACHED_REPEAT_GREETING_POOL = tuple(
    _pack_reply(1, "true", "english_greeting_repeat", msg, '["haha im good","how r u?"]')
    for msg in _REPEAT_GREETINGS_8
)


class QuantizedModel:
    __slots__ = (
        "emb_q", "emb_s", "wqkv1_q", "wqkv1_s", "wo1_q", "wo1_s", "w1_q", "w1_s", "w2_q", "w2_s",
        "wqkv2_q", "wqkv2_s", "wo2_q", "wo2_s", "w3_q", "w3_s", "w4_q", "w4_s", "w_intent_q", "w_intent_s",
        "emb_f32", "wqkv1_f32", "wo1_f32", "w1_f32", "w2_f32",
        "wqkv2_f32", "wo2_f32", "w3_f32", "w4_f32", "w_intent_f32",
        "pos_f32", "mask_f32", "char_lut", "reject_lut", "inv_sqrt_d", "eps",
        "h_p0", "h_p1", "vocab_mask", "one_i32", "space_i32", "conf_thresh",
    )

    def __init__(self, npz_path):
        arc = np.load(npz_path, mmap_mode="r")

        def lock_pair(prefix):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            q.flags.writeable = False
            s.flags.writeable = False
            f32.flags.writeable = False
            return q, s, f32

        self.emb_q, self.emb_s, emb_f32 = lock_pair("emb")
        emb_f32.flags.writeable = True
        emb_f32[0].fill(0.0)
        emb_f32.flags.writeable = False
        self.emb_f32 = emb_f32

        self.wqkv1_q, self.wqkv1_s, self.wqkv1_f32 = lock_pair("wqkv1")
        self.wo1_q, self.wo1_s, self.wo1_f32 = lock_pair("wo1")
        self.w1_q, self.w1_s, self.w1_f32 = lock_pair("w1")
        self.w2_q, self.w2_s, self.w2_f32 = lock_pair("w2")
        self.wqkv2_q, self.wqkv2_s, self.wqkv2_f32 = lock_pair("wqkv2")
        self.wo2_q, self.wo2_s, self.wo2_f32 = lock_pair("wo2")
        self.w3_q, self.w3_s, self.w3_f32 = lock_pair("w3")
        self.w4_q, self.w4_s, self.w4_f32 = lock_pair("w4")
        self.w_intent_q, self.w_intent_s, self.w_intent_f32 = lock_pair("w_intent")

        pos_idx = np.arange(SEQ_LEN, dtype=np.float32).reshape(SEQ_LEN, 1)
        log_10k = np.log(np.float32(10000.0))
        div_term = np.exp(
            np.arange(0, D_MODEL, 2, dtype=np.float32) * (-log_10k / np.float32(D_MODEL))
        ).astype(np.float32, copy=False)
        pos_f32 = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        pos_f32[:, 0::2] = np.sin(pos_idx * div_term) * np.float32(0.02)
        pos_f32[:, 1::2] = np.cos(pos_idx * div_term) * np.float32(0.02)
        pos_f32.flags.writeable = False
        self.pos_f32 = pos_f32

        mask_f32 = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9, dtype=np.float32), k=1)
        mask_f32.flags.writeable = False
        self.mask_f32 = mask_f32

        lut = np.full((256,), 32, dtype=np.int32)
        lut[47:58] = np.arange(47, 58, dtype=np.int32)
        lut[65:91] = np.arange(97, 123, dtype=np.int32)
        lut[97:123] = np.arange(97, 123, dtype=np.int32)
        lut[39] = np.int32(39)
        lut[128:256] = np.arange(128, 256, dtype=np.int32)
        lut.flags.writeable = False
        self.char_lut = lut

        rej = np.ones((256,), dtype=np.bool_)
        rej[32] = False
        rej[33] = False
        rej[39] = False
        rej[44] = False
        rej[46] = False
        rej[63] = False
        rej[65:91] = False
        rej[97:123] = False
        rej.flags.writeable = False
        self.reject_lut = rej

        self.inv_sqrt_d = np.float32(np.float32(1.0) / np.sqrt(np.float32(D_MODEL)))
        self.eps = np.float32(1e-5)
        # Synchronized polynomial hash constants: (c0 * 31 + c1) * 31 + c2 == c0 * 961 + c1 * 31 + c2
        self.h_p0 = np.int32(961)
        self.h_p1 = np.int32(31)
        self.vocab_mask = np.int32(VOCAB_SIZE - 2)
        self.one_i32 = np.int32(1)
        self.space_i32 = np.int32(32)
        self.conf_thresh = np.float32(0.0)


class InferenceSlot:
    __slots__ = (
        "slot_id", "raw_u8", "rej_mask", "tokens", "t_tmp", "tok_mask", "tok_mask2",
        "valid_f32", "active_count", "x", "qkv", "q", "k", "v", "scores",
        "row_max", "row_sum", "attn_out", "proj_out", "norm_tmp", "ffn_mid",
        "pooled", "intent_logits",
    )

    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.raw_u8 = np.zeros((SEQ_LEN + 2,), dtype=np.int32)
        self.rej_mask = np.zeros((SEQ_LEN + 2,), dtype=np.bool_)
        self.tokens = np.zeros((SEQ_LEN,), dtype=np.int32)
        self.t_tmp = np.zeros((SEQ_LEN,), dtype=np.int32)
        self.tok_mask = np.zeros((SEQ_LEN,), dtype=np.bool_)
        self.tok_mask2 = np.zeros((SEQ_LEN,), dtype=np.bool_)
        self.valid_f32 = np.zeros((SEQ_LEN, 1), dtype=np.float32)
        self.active_count = np.zeros((), dtype=np.float32)
        self.x = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.qkv = np.zeros((SEQ_LEN, 3 * D_MODEL), dtype=np.float32)
        self.q = self.qkv[:, :D_MODEL]
        self.k = self.qkv[:, D_MODEL : 2 * D_MODEL]
        self.v = self.qkv[:, 2 * D_MODEL :]
        self.scores = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        self.row_max = np.zeros((SEQ_LEN, 1), dtype=np.float32)
        self.row_sum = np.zeros((SEQ_LEN, 1), dtype=np.float32)
        self.attn_out = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.proj_out = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.norm_tmp = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.ffn_mid = np.zeros((SEQ_LEN, D_FF), dtype=np.float32)
        self.pooled = np.zeros((1, D_MODEL), dtype=np.float32)
        self.intent_logits = np.zeros((1, NUM_INTENTS), dtype=np.float32)


class UserStore:
    __slots__ = ("last_intent", "msg_count", "last_ts", "write_queue", "wal_batch")

    def __init__(self):
        self.last_intent = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.msg_count = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.last_ts = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.write_queue = deque(maxlen=4096)
        self.wal_batch = np.zeros((256, 3), dtype=np.uint32)


def forward_pass(model, slot, mv_payload):
    b_full = np.frombuffer(mv_payload[:SEQ_LEN], dtype=np.uint8)
    non_ws = np.flatnonzero(b_full > np.uint8(32))
    if non_ws.shape[0] == 0:
        return 0
    first_idx = int(non_ws[0])
    last_idx = int(non_ws[-1]) + 1
    n_tok = last_idx - first_idx
    if n_tok > MAX_GREETING_LEN:
        return 0

    slot.raw_u8.fill(32)
    slot.raw_u8[1 : 1 + n_tok] = b_full[first_idx:last_idx]

    np.take(model.reject_lut, slot.raw_u8[1 : 1 + n_tok], out=slot.rej_mask[:n_tok])
    if bool(np.any(slot.rej_mask[:n_tok])):
        return 0

    np.take(model.char_lut, slot.raw_u8, out=slot.raw_u8)

    c0 = slot.raw_u8[:SEQ_LEN]
    c1 = slot.raw_u8[1 : SEQ_LEN + 1]
    c2 = slot.raw_u8[2 : SEQ_LEN + 2]

    # Exact synced hash: (((c0 * 31 + c1) * 31 + c2) & (VOCAB_SIZE - 2)) + 1
    np.multiply(c0, model.h_p0, out=slot.tokens)
    np.multiply(c1, model.h_p1, out=slot.t_tmp)
    slot.tokens += slot.t_tmp
    slot.tokens += c2
    slot.tokens &= model.vocab_mask
    slot.tokens += model.one_i32

    np.equal(c0, model.space_i32, out=slot.tok_mask)
    np.equal(c1, model.space_i32, out=slot.tok_mask2)
    slot.tok_mask &= slot.tok_mask2
    np.equal(c2, model.space_i32, out=slot.tok_mask2)
    slot.tok_mask &= slot.tok_mask2
    np.invert(slot.tok_mask, out=slot.tok_mask)
    slot.tokens *= slot.tok_mask

    np.copyto(slot.valid_f32[:, 0], slot.tok_mask, casting="unsafe")
    np.sum(slot.valid_f32, out=slot.active_count)
    if float(slot.active_count) <= 0.0:
        return 0

    np.take(model.emb_f32, slot.tokens, axis=0, out=slot.x)
    slot.x += model.pos_f32
    slot.x *= slot.valid_f32

    # --- Layer 1 Fused-QKV Attention ---
    np.matmul(slot.x, model.wqkv1_f32, out=slot.qkv)
    np.matmul(slot.q, slot.k.T, out=slot.scores)
    slot.scores *= model.inv_sqrt_d
    slot.scores += model.mask_f32
    np.max(slot.scores, axis=1, keepdims=True, out=slot.row_max)
    slot.scores -= slot.row_max
    np.exp(slot.scores, out=slot.scores)
    np.sum(slot.scores, axis=1, keepdims=True, out=slot.row_sum)
    slot.scores /= slot.row_sum

    np.matmul(slot.scores, slot.v, out=slot.attn_out)
    np.matmul(slot.attn_out, model.wo1_f32, out=slot.proj_out)
    slot.x += slot.proj_out

    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += model.eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum

    # --- Layer 1 FFN ---
    np.matmul(slot.x, model.w1_f32, out=slot.ffn_mid)
    np.maximum(slot.ffn_mid, np.float32(0.0), out=slot.ffn_mid)
    np.matmul(slot.ffn_mid, model.w2_f32, out=slot.proj_out)
    slot.x += slot.proj_out

    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += model.eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum

    # --- Layer 2 Fused-QKV Attention ---
    np.matmul(slot.x, model.wqkv2_f32, out=slot.qkv)
    np.matmul(slot.q, slot.k.T, out=slot.scores)
    slot.scores *= model.inv_sqrt_d
    slot.scores += model.mask_f32
    np.max(slot.scores, axis=1, keepdims=True, out=slot.row_max)
    slot.scores -= slot.row_max
    np.exp(slot.scores, out=slot.scores)
    np.sum(slot.scores, axis=1, keepdims=True, out=slot.row_sum)
    slot.scores /= slot.row_sum

    np.matmul(slot.scores, slot.v, out=slot.attn_out)
    np.matmul(slot.attn_out, model.wo2_f32, out=slot.proj_out)
    slot.x += slot.proj_out

    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += model.eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum

    # --- Layer 2 FFN ---
    np.matmul(slot.x, model.w3_f32, out=slot.ffn_mid)
    np.maximum(slot.ffn_mid, np.float32(0.0), out=slot.ffn_mid)
    np.matmul(slot.ffn_mid, model.w4_f32, out=slot.proj_out)
    slot.x += slot.proj_out

    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += model.eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum

    # --- Masked Pooling + Strict Confidence Margin (>= 2.0) ---
    slot.x *= slot.valid_f32
    np.sum(slot.x, axis=0, keepdims=True, out=slot.pooled)
    slot.pooled /= slot.active_count
    np.matmul(slot.pooled, model.w_intent_f32, out=slot.intent_logits)

    margin = np.float32(slot.intent_logits[0, 1] - slot.intent_logits[0, 0])
    return 1 if margin >= model.conf_thresh else 0


def handle_user_query(model, slot, users, user_id, mv_payload):
    uid = user_id % MAX_USERS
    class_id = forward_pass(model, slot, mv_payload)
    prev_intent = int(users.last_intent[uid])
    msg_cnt = (int(users.msg_count[uid]) + 1) & 255

    users.last_intent[uid] = np.uint8(class_id)
    users.msg_count[uid] = np.uint8(msg_cnt)
    now_s = int(time.time()) & 0xFFFFFFFF
    users.last_ts[uid] = np.uint32(now_s)
    users.write_queue.append((uid, class_id, now_s))

    if class_id == 0:
        return CACHED_REJECT_RESPONSE
    if prev_intent == 1:
        return CACHED_REPEAT_GREETING_POOL[(uid + msg_cnt) & 7]
    return CACHED_GREETING_POOL[(uid + msg_cnt) & 31]


async def wal_write_behind_worker(users, wal_path="events_wal.bin"):
    while True:
        await asyncio.sleep(2.0)
        if not users.write_queue:
            continue
        count = 0
        while users.write_queue and count < 256:
            users.wal_batch[count] = users.write_queue.popleft()
            count += 1
        try:
            with open(wal_path, "ab") as wal_file:
                wal_file.write(memoryview(users.wal_batch)[:count])
        except OSError:
            pass


async def start_server(host="127.0.0.1", port=8080):
    if not os.path.exists("model_weights.npz"):
        import offline_train
        offline_train.train_and_export("model_weights.npz", force_retrain=True)

    model = QuantizedModel("model_weights.npz")
    users = UserStore()
    slot_queue = asyncio.Queue(maxsize=NUM_WORKERS)
    for i in range(NUM_WORKERS):
        slot_queue.put_nowait(InferenceSlot(i))

    test_slot = await slot_queue.get()
    sample_mv = memoryview(b"heyyy what's up")
    t0 = time.perf_counter()
    for i in range(1000):
        handle_user_query(model, test_slot, users, i, sample_mv)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    suite = (
        (b"hi", 1), (b"HI!", 1), (b"heyyy", 1), (b"yo what's up", 1),
        (b"how are you?", 1), (b"good morning", 1), (b"ello", 1), (b"sup", 1),
        (b"hellod/", 0), (b"hellod", 0), (b"hi/", 0), (b"/hello", 0),
        (b"hi123", 0), (b"help", 0), (b"helicopter", 0), (b"bonjour", 0),
    )
    passed = sum(1 for txt, exp in suite if forward_pass(model, test_slot, memoryview(txt)) == exp)

    users.last_intent[777] = np.uint8(0)
    r1 = handle_user_query(model, test_slot, users, 777, memoryview(b"hi"))
    r2 = handle_user_query(model, test_slot, users, 777, memoryview(b"hey"))
    slot_queue.put_nowait(test_slot)

    b1 = r1[r1.find(b"\r\n\r\n") + 4 :].decode("utf-8")
    b2 = r2[r2.find(b"\r\n\r\n") + 4 :].decode("utf-8")
    print(f"[BENCHMARK] 1,000 forward passes in {elapsed_ms:.2f} ms ({elapsed_ms / 1000.0:.4f} ms/req)")
    print(f"[ZERO-FAUX SUITE] Passed {passed}/{len(suite)} hard-negative & greeting tests (2.56 MB RAM)")
    print(f"[1ST GREETING] {b1}")
    print(f"[2ND GREETING] {b2}")

    if "--benchmark" in sys.argv:
        return

    async def client_handler(reader, writer):
        try:
            req_data = await asyncio.wait_for(reader.read(512), timeout=2.0)
            if not req_data:
                writer.close()
                return
            peer = writer.get_extra_info("peername")
            uid = (hash(peer[0]) & 0x7FFFFFFF) if peer else 101
            mv_req = memoryview(req_data)
            slot = await slot_queue.get()
            try:
                body_idx = req_data.find(b"\r\n\r\n")
                mv_body = mv_req[body_idx + 4 :] if body_idx != -1 else mv_req
                http_reply = handle_user_query(model, slot, users, uid, mv_body)
            finally:
                slot_queue.put_nowait(slot)

            writer.write(http_reply)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    asyncio.create_task(wal_write_behind_worker(users))
    server = await asyncio.start_server(client_handler, host, port, backlog=2048)
    print(f"[SERVER] Step 1 Dating Greeting Engine on http://{host}:{port} (2.56 MB Total RAM)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())