# FILE: offline_train.py
# PURPOSE: Subword-trigram Transformer training & int8 export to model_weights.npz
# RULES: Zero float64/int64, masked non-padding pooling, PyArmor compatible (< 220 lines)

import numpy as np

VOCAB_SIZE = 512
SEQ_LEN = 64
D_MODEL = 128
D_FF = 256
NUM_INTENTS = 8

TRAINING_SET = (
    (b"hello hi hey good morning welcome", 0),
    (b"hi there hello how are you doing", 0),
    (b"salam marhba ahla bik kifech", 0),
    (b"bonjour salut coucou ca va bien", 0),
    (b"price cost how much is the price", 1),
    (b"hello how much is the price please", 1),
    (b"what is the pricing plan and cost", 1),
    (b"bonjour quel est le tarif et prix svp", 1),
    (b"qadeh el soum as3ar ishtirak", 1),
    (b"help me support problem issue", 2),
    (b"i need help from customer support", 2),
    (b"aide moi probleme support technique", 2),
    (b"3awenni 3andi mochkla da3m", 2),
    (b"where is my order status track shipping", 3),
    (b"track my package order number", 3),
    (b"suivre mon colis statut de commande", 3),
    (b"win talabiya tatabo3", 3),
    (b"i want a refund cancel my subscription", 4),
    (b"money back guarantee refund request", 4),
    (b"remboursement annuler mon abonnement retour", 4),
    (b"istirja3 flousi nheb nrajja3", 4),
    (b"api error bug server crash technical", 5),
    (b"system down connection error bug", 5),
    (b"erreur technique bug serveur panne", 5),
    (b"what are your open hours schedule time", 6),
    (b"when are you open working hours", 6),
    (b"quels sont vos horaires ouverture", 6),
    (b"waqtech mahloulin sa3at el 3amal", 6),
    (b"bye goodbye see you thank you", 7),
    (b"thanks a lot bye have a good day", 7),
    (b"merci beaucoup au revoir a bientot", 7),
    (b"shukran barcha bslama", 7),
)


def vector_subword_tokenize(raw_bytes_u8_mat, lengths):
    upper_mask = (raw_bytes_u8_mat >= np.int32(65)) & (raw_bytes_u8_mat <= np.int32(90))
    lower_mat = raw_bytes_u8_mat + upper_mask.astype(np.int32) * np.int32(32)

    v0 = lower_mat[:, :SEQ_LEN]
    v1 = lower_mat[:, 1 : SEQ_LEN + 1]
    v2 = lower_mat[:, 2 : SEQ_LEN + 2]

    tokens = ((v0 * np.int32(37)) + (v1 * np.int32(13)) + v2) & np.int32(511)
    zero_mask = (tokens == np.int32(0)).astype(np.int32)
    tokens += zero_mask

    col_idx = np.arange(SEQ_LEN, dtype=np.int32).reshape(1, SEQ_LEN)
    valid_mask = (col_idx < lengths.reshape(-1, 1)).astype(np.int32)
    return tokens * valid_mask, valid_mask.astype(np.float32)


def build_training_tensors():
    n = len(TRAINING_SET)
    # Fill with ASCII space (32) so end-of-string words share identical boundary trigrams
    padded_u8 = np.full((n, SEQ_LEN + 2), 32, dtype=np.int32)
    lengths = np.zeros((n,), dtype=np.int32)
    labels = np.zeros((n,), dtype=np.int32)

    for i, (text_b, intent_id) in enumerate(TRAINING_SET):
        b_arr = np.frombuffer(text_b[:SEQ_LEN], dtype=np.uint8)
        length = b_arr.shape[0]
        padded_u8[i, :length] = b_arr
        lengths[i] = np.int32(length)
        labels[i] = np.int32(intent_id)

    x_tok, valid_mask_f32 = vector_subword_tokenize(padded_u8, lengths)
    y_intent_oh = np.eye(NUM_INTENTS, dtype=np.float32)[labels]
    return x_tok, valid_mask_f32, y_intent_oh, labels


def quantize_to_int8(w_f32):
    max_abs = np.max(np.abs(w_f32))
    scale = (
        np.float32(max_abs / np.float32(127.0))
        if max_abs > np.float32(1e-8)
        else np.float32(1.0 / 127.0)
    )
    scaled = np.divide(w_f32, scale, dtype=np.float32)
    rounded = np.rint(scaled, dtype=np.float32)
    clipped = np.clip(rounded, np.float32(-127.0), np.float32(127.0), dtype=np.float32)
    return clipped.astype(np.int8), np.array([scale], dtype=np.float32)


def train_and_export(output_path="model_weights.npz"):
    x_np, mask_np, yi_oh_np, labels_np = build_training_tensors()
    rng = np.random.default_rng(42)

    def init_mat(rows, cols):
        limit = np.sqrt(np.float32(6.0) / np.float32(rows + cols))
        mat = rng.random((rows, cols), dtype=np.float32)
        mat *= np.float32(2.0) * limit
        mat -= limit
        return mat

    shapes = (
        (VOCAB_SIZE, D_MODEL),
        (D_MODEL, D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_MODEL),
        (D_MODEL, D_FF), (D_FF, D_MODEL),
        (D_MODEL, D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_MODEL),
        (D_MODEL, D_FF), (D_FF, D_MODEL),
        (D_MODEL, NUM_INTENTS),
    )
    mats = [init_mat(r, c) for r, c in shapes]
    mats[0][0] = np.float32(0.0)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    params = tuple(nn.Parameter(torch.from_numpy(m).to(torch.float32)) for m in mats)
    opt = torch.optim.AdamW(params, lr=1.5e-2, weight_decay=1e-4)

    x_t = torch.from_numpy(x_np).to(torch.int32)
    valid_m = torch.from_numpy(mask_np).to(torch.float32).unsqueeze(-1)
    valid_counts = torch.clamp(torch.sum(valid_m, dim=1), min=1.0)
    yi_oh = torch.from_numpy(yi_oh_np).to(torch.float32)

    pos_idx = torch.arange(SEQ_LEN, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, D_MODEL, 2, dtype=torch.float32)
        * (-float(np.log(np.float32(10000.0))) / float(D_MODEL))
    )
    pos_enc = torch.zeros((SEQ_LEN, D_MODEL), dtype=torch.float32)
    pos_enc[:, 0::2] = torch.sin(pos_idx * div) * 0.05
    pos_enc[:, 1::2] = torch.cos(pos_idx * div) * 0.05
    mask = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), -1e9, dtype=torch.float32), diagonal=1)
    scale_d = float(np.float32(1.0) / np.sqrt(np.float32(D_MODEL)))

    def rms_norm(tensor_x):
        var = torch.mean(tensor_x * tensor_x, dim=-1, keepdim=True)
        return tensor_x / torch.sqrt(var + 1e-5)

    for _ in range(300):
        opt.zero_grad()
        with torch.no_grad():
            params[0].data[0].zero_()

        h = F.embedding(x_t, params[0]) + pos_enc
        attn1 = F.softmax(((h @ params[1]) @ (h @ params[2]).transpose(-2, -1)) * scale_d + mask, dim=-1)
        h = rms_norm(h + (attn1 @ (h @ params[3])) @ params[4])
        h = rms_norm(h + F.relu(h @ params[5]) @ params[6])

        attn2 = F.softmax(((h @ params[7]) @ (h @ params[8]).transpose(-2, -1)) * scale_d + mask, dim=-1)
        h = rms_norm(h + (attn2 @ (h @ params[9])) @ params[10])
        h = rms_norm(h + F.relu(h @ params[11]) @ params[12])

        # Masked average pooling over ONLY active non-padding tokens
        pooled = torch.sum(h * valid_m, dim=1) / valid_counts
        logits_i = pooled @ params[13]

        loss = -torch.sum(yi_oh * F.log_softmax(logits_i, dim=-1), dim=-1).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        params[0].data[0].zero_()
        logits_np = logits_i.detach().cpu().numpy().astype(np.float32, copy=False)
        row_max = np.max(logits_np, axis=1, keepdims=True)
        idx_grid = np.arange(NUM_INTENTS, dtype=np.int32).reshape(1, NUM_INTENTS)
        preds = np.max((logits_np == row_max) * idx_grid, axis=1)
        acc = float(np.mean(preds == labels_np, dtype=np.float32)) * 100.0
        print(f"[TRAIN] Training Accuracy across all 32 multilingual phrases: {acc:.1f}%")

    trained = [p.detach().cpu().numpy().astype(np.float32, copy=False) for p in params]
    q_pairs = [quantize_to_int8(m) for m in trained]
    q_flat = {}
    names = (
        "emb", "wq1", "wk1", "wv1", "wo1", "w1", "w2",
        "wq2", "wk2", "wv2", "wo2", "w3", "w4", "w_intent",
    )
    for idx, name in enumerate(names):
        q_flat[f"{name}_q"] = q_pairs[idx][0]
        q_flat[f"{name}_s"] = q_pairs[idx][1]

    np.savez(output_path, **q_flat)
    print(f"[EXPORT] Saved quantized int8 model to '{output_path}' (321.06 KB in RAM)")


if __name__ == "__main__":
    train_and_export()