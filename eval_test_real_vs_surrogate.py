#!/usr/bin/env python3
import argparse
import numpy as np



def accuracy_from_scores(y_true, scores, prob_threshold=0.7):
    """
    Accuracy using a probability threshold.
    prob_threshold=0.7 corresponds to logit ≈ 0.8473
    """
    logit_threshold = np.log(prob_threshold / (1.0 - prob_threshold))
    y_pred = (scores >= logit_threshold).astype(np.int64)
    return float((y_pred == y_true).mean())

# ---------- IO ----------
def load_emb_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            pairs.append((a, b))
    return pairs

# ---------- Surrogate reconstruction ----------
def normalized_layer_coords(L: int) -> np.ndarray:
    if L <= 1:
        return np.zeros((L,), dtype=np.float32)
    l = np.arange(L, dtype=np.float32)
    return (2.0 * (l / float(L - 1)) - 1.0).astype(np.float32)

def eval_poly_ascending(coeffs_Kd1: np.ndarray, t: float) -> np.ndarray:
    deg = coeffs_Kd1.shape[1] - 1
    powers = np.array([t**j for j in range(deg + 1)], dtype=np.float32)
    return coeffs_Kd1 @ powers

def reconstruct_embedding(coeffs_Kd1, mean, comps, layer_idx: int, L: int) -> np.ndarray:
    t = float(normalized_layer_coords(L)[layer_idx])
    z = eval_poly_ascending(coeffs_Kd1, t)
    x = (z[None, :] @ comps).reshape(-1) + mean
    return x.astype(np.float32)

# ---------- Features ----------
def hadamard(a, b):
    return a * b

def build_Xy_real_filtered(emb_dict, pos_pairs, neg_pairs, layer):
    X_list, y_list = [], []
    skipped = 0

    for p, q in pos_pairs:
        if p not in emb_dict or q not in emb_dict:
            skipped += 1; continue
        X_list.append(hadamard(
            emb_dict[p][layer].astype(np.float32),
            emb_dict[q][layer].astype(np.float32)
        ))
        y_list.append(1)

    for p, q in neg_pairs:
        if p not in emb_dict or q not in emb_dict:
            skipped += 1; continue
        X_list.append(hadamard(
            emb_dict[p][layer].astype(np.float32),
            emb_dict[q][layer].astype(np.float32)
        ))
        y_list.append(0)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, skipped

def build_Xy_surrogate_filtered(coeffs_npz, pos_pairs, neg_pairs, layer):
    d = np.load(coeffs_npz, allow_pickle=True)

    pids_raw = d["pids"]
    pids = np.array(
        [p.decode("utf-8") if isinstance(p, (bytes, np.bytes_)) else str(p)
         for p in pids_raw],
        dtype=object
    )
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    mean = d["pca_mean"].astype(np.float32)
    comps = d["pca_components"].astype(np.float32)
    coeffs = d["coeffs"].astype(np.float32)
    L = int(d["L"])

    def emb(pid):
        return reconstruct_embedding(
            coeffs[pid_to_idx[pid]], mean, comps, layer, L
        )

    X_list, y_list = [], []
    skipped = 0

    for p, q in pos_pairs:
        if p not in pid_to_idx or q not in pid_to_idx:
            skipped += 1; continue
        X_list.append(hadamard(emb(p), emb(q)))
        y_list.append(1)

    for p, q in neg_pairs:
        if p not in pid_to_idx or q not in pid_to_idx:
            skipped += 1; continue
        X_list.append(hadamard(emb(p), emb(q)))
        y_list.append(0)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, skipped

# ---------- Metrics / classifier ----------
def auroc(y_true, scores):
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    pos = (y_true == 1)
    n_pos = int(pos.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))



def train_logreg_gd(X, y, l2=1e-4, iters=300, lr=0.1):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd

    w = np.zeros(Xs.shape[1], dtype=np.float32)
    b = 0.0
    y = y.astype(np.float32)

    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = (Xs.T @ (p - y)) / len(y) + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw.astype(np.float32)
        b -= lr * gb

    return (w, b, mu.astype(np.float32), sd.astype(np.float32))

def predict_scores(X, model):
    w, b, mu, sd = model
    return ((X - mu) / sd) @ w + b

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--test_coeffs_npz", required=True)
    ap.add_argument("--train_pos", required=True)
    ap.add_argument("--train_neg", required=True)
    ap.add_argument("--test_pos", required=True)
    ap.add_argument("--test_neg", required=True)
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    train_emb = load_emb_npz(args.train_npz)
    test_emb = load_emb_npz(args.test_npz)

    tr_pos = load_pairs(args.train_pos)
    tr_neg = load_pairs(args.train_neg)
    te_pos = load_pairs(args.test_pos)
    te_neg = load_pairs(args.test_neg)

    layer = args.layer
    print(f"[info] evaluating TEST at layer {layer}", flush=True)

    # Train on TRAIN REAL
    Xtr, ytr, skip_tr = build_Xy_real_filtered(train_emb, tr_pos, tr_neg, layer)
    model = train_logreg_gd(Xtr, ytr)

    # Test REAL
    Xte_real, yte, skip_te_r = build_Xy_real_filtered(test_emb, te_pos, te_neg, layer)
    scores_real = predict_scores(Xte_real, model)
    auc_real = auroc(yte, scores_real)
    acc_real = accuracy_from_scores(yte, scores_real)

    # Test SURROGATE
    Xte_surr, yte2, skip_te_s = build_Xy_surrogate_filtered(
        args.test_coeffs_npz, te_pos, te_neg, layer
    )
    scores_surr = predict_scores(Xte_surr, model)
    auc_surr = auroc(yte2, scores_surr)
    acc_surr = accuracy_from_scores(yte2, scores_surr)

    print(
        f"[TEST] "
        f"AUROC(real)={auc_real:.4f}  "
        f"AUROC(surrogate)={auc_surr:.4f}  "
        f"ACC(real)={acc_real:.4f}  "
        f"ACC(surrogate)={acc_surr:.4f}"
    )
    print(f"[skipped pairs] train={skip_tr} test_real={skip_te_r} test_surr={skip_te_s}")

if __name__ == "__main__":
    main()
