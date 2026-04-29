#!/usr/bin/env python3
import argparse
import numpy as np

def load_emb_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

def normalized_layer_coords(L: int) -> np.ndarray:
    if L <= 1:
        return np.zeros((L,), dtype=np.float32)
    l = np.arange(L, dtype=np.float32)
    return (2.0 * (l / float(L - 1)) - 1.0).astype(np.float32)

def pca_transform(X, mean, components):
    return (X.astype(np.float32, copy=False) - mean) @ components.T  # [L,K]

def fit_poly_coeffs(pooled_LD, mean, components, degree):
    L, _ = pooled_LD.shape
    Z = pca_transform(pooled_LD, mean, components)  # [L,K]
    t = normalized_layer_coords(L)                  # [L]
    K = Z.shape[1]
    coeffs = np.zeros((K, degree + 1), dtype=np.float32)
    for k in range(K):
        c_desc = np.polyfit(t, Z[:, k], deg=degree)     # [cd..c0]
        coeffs[k] = c_desc[::-1].astype(np.float32)     # [c0..cd]
    return coeffs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_model", required=True, help="cubic_model_train_K64_deg3.npz")
    ap.add_argument("--val_npz", required=True, help="emb_val_out/val_pooled_by_layer.npz")
    ap.add_argument("--out_npz", required=True, help="output coeffs npz for val proteins")
    args = ap.parse_args()

    m = np.load(args.train_model, allow_pickle=True)
    mean = m["pca_mean"].astype(np.float32)
    comps = m["pca_components"].astype(np.float32)
    degree = int(m["degree"]) if "degree" in m.files else 3

    pooled = load_emb_npz(args.val_npz)
    pids = np.array(sorted(pooled.keys()))
    L, D = pooled[pids[0]].shape
    K = comps.shape[0]

    print(f"[val] proteins={len(pids)} L={L} D={D}  using train PCA K={K} degree={degree}")

    coeffs = np.zeros((len(pids), K, degree + 1), dtype=np.float32)
    for i, pid in enumerate(pids):
        coeffs[i] = fit_poly_coeffs(pooled[pid], mean, comps, degree)
        if (i + 1) % 200 == 0:
            print(f"[fit] {i+1}/{len(pids)}")

    np.savez(
        args.out_npz,
        pids=pids,
        L=np.int32(L),
        D=np.int32(D),
        n_pcs=np.int32(K),
        degree=np.int32(degree),
        pca_mean=mean,
        pca_components=comps,
        coeffs=coeffs,
    )
    print(f"[save] wrote {args.out_npz}")

if __name__ == "__main__":
    main()
