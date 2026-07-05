#!/usr/bin/env python3
"""
Reconstruct surrogate embeddings for all proteins from polynomial coefficient
files, producing an NPZ in the same format as the original embeddings:
    pid -> array[L, D]

Usage:
    python reconstruct_surrogate.py \
        --coeffs_files surrogate_deeploc_35M_K64_deg3.npz \
                       surrogate_deeploc_35M_test_coeffs_K64_deg3.npz \
        --out_npz      deeploc_surrogate_35M_K64_deg3.npz
"""
import argparse
import numpy as np


def normalized_layer_coords(L: int) -> np.ndarray:
    if L <= 1:
        return np.zeros((L,), dtype=np.float32)
    l = np.arange(L, dtype=np.float32)
    return (2.0 * (l / float(L - 1)) - 1.0).astype(np.float32)


def reconstruct_protein(coeffs_KD1, mean, components, L):
    """coeffs_KD1: [K, deg+1] -> reconstructed [L, D]."""
    deg = coeffs_KD1.shape[1] - 1
    t = normalized_layer_coords(L)                                # [L]
    powers = np.stack([t ** j for j in range(deg + 1)], axis=1)   # [L, deg+1]
    Z = coeffs_KD1 @ powers.T                                     # [K, L]
    Z = Z.T                                                       # [L, K]
    X = Z @ components + mean                                     # [L, D]
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs_files", nargs="+", required=True,
                    help="one or more coeffs NPZ files (e.g. train + test)")
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    out = {}
    pca_mean = pca_components = None
    L_global = None

    for path in args.coeffs_files:
        print(f"[load] {path}", flush=True)
        d = np.load(path, allow_pickle=True)
        pids = d["pids"]
        coeffs = d["coeffs"]                # [N, K, deg+1]
        mean = d["pca_mean"].astype(np.float32)
        comps = d["pca_components"].astype(np.float32)
        L = int(d["L"])
        if pca_mean is None:
            pca_mean, pca_components, L_global = mean, comps, L
        else:
            assert np.allclose(mean, pca_mean), f"PCA mean mismatch in {path}"
            assert np.allclose(comps, pca_components), f"PCA components mismatch in {path}"
            assert L == L_global, f"L mismatch in {path}"

        for i, pid in enumerate(pids):
            pid_str = str(pid)
            if pid_str in out:
                print(f"[warn] duplicate pid {pid_str}, overwriting", flush=True)
            out[pid_str] = reconstruct_protein(coeffs[i], mean, comps, L)
            if (i + 1) % 2000 == 0:
                print(f"[recon] {i+1}/{len(pids)} from {path}", flush=True)

    print(f"[save] {len(out)} proteins -> {args.out_npz}", flush=True)
    np.savez(args.out_npz, **out)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()