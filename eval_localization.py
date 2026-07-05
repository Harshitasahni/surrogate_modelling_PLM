#!/usr/bin/env python3
"""
Linear probe evaluation on DeepLoc 1.0 subcellular localization.

For each layer:
  - Stratified 90/10 train/val carve.
  - Grid over C; pick best C by val macro-AUROC.
  - Refit on full train with best C; report 6 test metrics.

Final layer selection is by val macro-AUROC (test is touched only once
per layer for reporting). The chosen layer is printed at the end.

Usage:
    python eval_localization.py \
        --embeddings deeploc_pooled_35M.npz \
        --labels deeploc/labels.csv \
        --out results_real_35M.csv

    # single layer:
    python eval_localization.py \
        --embeddings deeploc_pooled_35M.npz \
        --labels deeploc/labels.csv \
        --layer 8 \
        --out results_real_35M_layer8.csv
"""
import argparse
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)

C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_aligned(emb_path, labels_path):
    """Load embeddings + labels, align by protein_id."""
    print(f"[load] embeddings: {emb_path}", flush=True)
    emb = np.load(emb_path)
    emb_pids = set(emb.files)
    print(f"[load] {len(emb_pids)} proteins in embeddings", flush=True)

    print(f"[load] labels: {labels_path}", flush=True)
    df = pd.read_csv(labels_path)
    df = df[df["protein_id"].isin(emb_pids)].reset_index(drop=True)
    print(f"[load] {len(df)} aligned proteins", flush=True)

    sample_shape = emb[df["protein_id"].iloc[0]].shape  # (L, D)
    X = np.empty((len(df), *sample_shape), dtype=np.float32)
    for i, pid in enumerate(df["protein_id"].values):
        X[i] = emb[pid]
    print(f"[load] X shape: {X.shape}", flush=True)

    le = LabelEncoder()
    y = le.fit_transform(df["location"].values)
    print(f"[load] {len(le.classes_)} classes: {list(le.classes_)}", flush=True)

    splits = df["split"].values
    n_train = int((splits == "train").sum())
    n_test  = int((splits == "test").sum())
    print(f"[load] train={n_train} test={n_test}", flush=True)

    return X, y, splits, le.classes_


def macro_metrics(y_true, preds, probs, n_classes):
    """Return all 6 metrics: matches the PPI eval set, multi-class extensions."""
    y_oh = np.zeros((len(y_true), n_classes), dtype=np.int8)
    y_oh[np.arange(len(y_true)), y_true] = 1

    return {
        "accuracy":  accuracy_score(y_true, preds),
        "auroc":     roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "auprc":     average_precision_score(y_oh, probs, average="weighted"),
        "f1":        f1_score(y_true, preds, average="weighted"),
        "precision": precision_score(y_true, preds, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, preds, average="weighted", zero_division=0),
    }


def evaluate_layer(X_layer, y, splits, n_classes, seed=42):
    """Train probe, return val macro-AUROC (for layer/C selection) + full test metrics."""
    train_mask = (splits == "train")
    test_mask  = (splits == "test")

    X_tr_full = X_layer[train_mask]
    y_tr_full = y[train_mask]
    X_te      = X_layer[test_mask]
    y_te      = y[test_mask]

    # stratified 90/10 train/val carve
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=0.1, stratify=y_tr_full, random_state=seed,
    )

    # standardize using carve-train stats
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # C grid by val macro-AUROC
    best_C, best_val_auroc = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        val_probs = clf.predict_proba(X_val_s)
        val_auroc = roc_auc_score(y_val, val_probs, multi_class="ovr", average="macro")
        if val_auroc > best_val_auroc:
            best_val_auroc, best_C = val_auroc, C

    # refit on FULL train with best C, eval on test
    scaler_full = StandardScaler().fit(X_tr_full)
    X_tr_full_s = scaler_full.transform(X_tr_full)
    X_te_s      = scaler_full.transform(X_te)

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs", n_jobs=-1)
    clf.fit(X_tr_full_s, y_tr_full)

    probs = clf.predict_proba(X_te_s)
    preds = probs.argmax(axis=1)
    test_metrics = macro_metrics(y_te, preds, probs, n_classes)

    return {
        "best_C": best_C,
        "val_auroc": best_val_auroc,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="NPZ: pid -> [L, D]")
    ap.add_argument("--labels", required=True, help="CSV: protein_id, location, split")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--layer", type=int, default=None,
                    help="single layer (default: scan all layers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, splits, classes = load_aligned(args.embeddings, args.labels)
    n_layers, _ = X.shape[1], X.shape[2]
    n_classes = len(classes)

    layers = [args.layer] if args.layer is not None else list(range(n_layers))

    rows = []
    for ell in layers:
        print(f"\n[layer {ell}] training...", flush=True)
        res = evaluate_layer(X[:, ell, :], y, splits, n_classes, seed=args.seed)
        row = {"layer": ell, **res}
        rows.append(row)
        print(
            f"[layer {ell}] best_C={res['best_C']} "
            f"val_auroc={res['val_auroc']:.4f} | "
            f"test acc={res['test_accuracy']:.4f} "
            f"auroc={res['test_auroc']:.4f} "
            f"auprc={res['test_auprc']:.4f} "
            f"f1={res['test_f1']:.4f}",
            flush=True,
        )

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {args.out}", flush=True)

    # final layer selection by VAL auroc
    if len(rows) > 1:
        best = max(rows, key=lambda r: r["val_auroc"])
        print("\n" + "=" * 60)
        print(f"BEST LAYER (by val macro-AUROC): layer {best['layer']}")
        print(f"  val_auroc:      {best['val_auroc']:.4f}")
        print(f"  test accuracy:  {best['test_accuracy']:.4f}")
        print(f"  test auroc:     {best['test_auroc']:.4f}")
        print(f"  test auprc:     {best['test_auprc']:.4f}")
        print(f"  test f1:        {best['test_f1']:.4f}")
        print(f"  test precision: {best['test_precision']:.4f}")
        print(f"  test recall:    {best['test_recall']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
    
    
'''
python eval_localization.py --embeddings deeploc_pooled_35M.npz --labels deeploc/labels.csv --out results_real_35M.csv
'''  