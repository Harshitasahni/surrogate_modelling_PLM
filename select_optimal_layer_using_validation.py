#!/usr/bin/env python3
import argparse
import numpy as np

# ---------------- IO ----------------
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

# ---------------- Surrogate reconstruction ----------------
def normalized_layer_coords(L: int) -> np.ndarray:
    if L <= 1:
        return np.zeros((L,), dtype=np.float32)
    l = np.arange(L, dtype=np.float32)
    return (2.0 * (l / float(L - 1)) - 1.0).astype(np.float32)

def eval_poly_ascending(coeffs_Kd1: np.ndarray, t: float) -> np.ndarray:
    deg = coeffs_Kd1.shape[1] - 1
    powers = np.array([t**j for j in range(deg + 1)], dtype=np.float32)
    return coeffs_Kd1 @ powers  # [K]

def reconstruct_embedding(coeffs_Kd1, mean, comps, layer_idx: int, L: int) -> np.ndarray:
    t = float(normalized_layer_coords(L)[layer_idx])
    z = eval_poly_ascending(coeffs_Kd1, t)  # [K]
    x = (z[None, :] @ comps).reshape(-1) + mean
    return x.astype(np.float32)

# ---------------- Features ----------------
def hadamard(a, b):
    return a * b

def build_Xy_real_filtered(emb_dict, pos_pairs, neg_pairs, layer):
    X_list, y_list, skipped = [], [], 0
    for p, q in pos_pairs:
        if p not in emb_dict or q not in emb_dict: skipped += 1; continue
        X_list.append(hadamard(emb_dict[p][layer].astype(np.float32),
                               emb_dict[q][layer].astype(np.float32)))
        y_list.append(1)
    for p, q in neg_pairs:
        if p not in emb_dict or q not in emb_dict: skipped += 1; continue
        X_list.append(hadamard(emb_dict[p][layer].astype(np.float32),
                               emb_dict[q][layer].astype(np.float32)))
        y_list.append(0)
    if not X_list:
        raise RuntimeError("No valid pairs left after filtering missing proteins (REAL).")
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64), skipped

def build_Xy_surrogate_filtered(coeffs_npz, pos_pairs, neg_pairs, layer):
    d = np.load(coeffs_npz, allow_pickle=True)
    pids = np.array([p.decode("utf-8") if isinstance(p, (bytes, np.bytes_)) else str(p)
                     for p in d["pids"]], dtype=object)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    mean   = d["pca_mean"].astype(np.float32)
    comps  = d["pca_components"].astype(np.float32)
    coeffs = d["coeffs"].astype(np.float32)
    L      = int(d["L"])

    def has(pid): return pid in pid_to_idx
    def emb(pid): return reconstruct_embedding(coeffs[pid_to_idx[pid]], mean, comps, layer, L)

    X_list, y_list, skipped = [], [], 0
    for p, q in pos_pairs:
        if not has(p) or not has(q): skipped += 1; continue
        X_list.append(hadamard(emb(p), emb(q))); y_list.append(1)
    for p, q in neg_pairs:
        if not has(p) or not has(q): skipped += 1; continue
        X_list.append(hadamard(emb(p), emb(q))); y_list.append(0)
    if not X_list:
        raise RuntimeError("No valid pairs left after filtering missing proteins (SURROGATE).")
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64), skipped

# ---------------- Metrics / classifier ----------------
def auprc(y_true, scores):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (y_sorted.sum() + 1e-9)
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))

def auroc(y_true, scores):
    y_true  = y_true.astype(np.int64)
    scores  = scores.astype(np.float64)
    order   = np.argsort(scores)
    ranks   = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    pos     = (y_true == 1)
    n_pos   = int(pos.sum())
    n_neg   = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def train_logreg_gd(X, y, l2=1e-4, iters=300, lr=0.1):
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    w  = np.zeros(Xs.shape[1], dtype=np.float32); b = 0.0
    y  = y.astype(np.float32)
    for _ in range(iters):
        p   = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        gw  = (Xs.T @ (p - y)) / len(y) + l2 * w
        gb  = float(np.mean(p - y))
        w  -= lr * gw.astype(np.float32); b -= lr * gb
    return (w, b, mu.astype(np.float32), sd.astype(np.float32))

def predict_scores(X, model):
    w, b, mu, sd = model
    return (X - mu) / sd @ w + b

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz",      required=True)
    ap.add_argument("--val_npz",        required=True)
    ap.add_argument("--val_coeffs_npz", required=True)
    ap.add_argument("--train_pos",      required=True)
    ap.add_argument("--train_neg",      required=True)
    ap.add_argument("--val_pos",        required=True)
    ap.add_argument("--val_neg",        required=True)
    ap.add_argument("--mode", choices=["scan_layers", "one_layer"], default="scan_layers")
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()

    train_emb = load_emb_npz(args.train_npz)
    val_emb   = load_emb_npz(args.val_npz)
    tr_pos = load_pairs(args.train_pos); tr_neg = load_pairs(args.train_neg)
    va_pos = load_pairs(args.val_pos);   va_neg = load_pairs(args.val_neg)

    L      = train_emb[next(iter(train_emb))].shape[0]
    layers = list(range(L)) if args.mode == "scan_layers" else [args.layer]

    print(f"[info] L={L} train_pairs={len(tr_pos)+len(tr_neg)} val_pairs={len(va_pos)+len(va_neg)}", flush=True)

    best_real = (-1.0, None)
    best_surr = (-1.0, None)   # ← track best surrogate separately

    for layer in layers:
        Xtr, ytr, skip_tr   = build_Xy_real_filtered(train_emb, tr_pos, tr_neg, layer)
        model               = train_logreg_gd(Xtr, ytr, l2=1e-4, iters=300, lr=0.1)

        Xva_real, yva, skip_va   = build_Xy_real_filtered(val_emb, va_pos, va_neg, layer)
        scores_real              = predict_scores(Xva_real, model)
        auc_real                 = auroc(yva, scores_real)

        Xva_surr, yva2, skip_vs  = build_Xy_surrogate_filtered(args.val_coeffs_npz, va_pos, va_neg, layer)
        if len(yva2) != len(yva):
            print(f"[warn] label length mismatch: real={len(yva)} surr={len(yva2)}", flush=True)
        else:
            assert np.all(yva2 == yva)

        scores_surr = predict_scores(Xva_surr, model)
        auc_surr    = auroc(yva2, scores_surr)

        print(
            f"layer={layer:2d}  "
            f"val_AUROC(real)={auc_real:.4f}  "
            f"val_AUROC(surr)={auc_surr:.4f}  "
            f"skip_train={skip_tr} skip_val_real={skip_va} skip_val_surr={skip_vs}",
            flush=True
        )

        if auc_real > best_real[0]: best_real = (auc_real, layer)
        if auc_surr > best_surr[0]: best_surr = (auc_surr, layer)

    if args.mode == "scan_layers":
        print(f"\n[best_real_layer] layer={best_real[1]}  val_AUROC(real)={best_real[0]:.4f}", flush=True)
        print(f"[best_surr_layer] layer={best_surr[1]}  val_AUROC(surr)={best_surr[0]:.4f}", flush=True)

if __name__ == "__main__":
    main()