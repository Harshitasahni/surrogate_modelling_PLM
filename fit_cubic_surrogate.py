#!/usr/bin/env python3
import argparse
import numpy as np

def load_npz_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # EACH ID has embeddings for each layer [L,D]
    emb = {k: data[k] for k in data.files}
    return emb

def fit_pca_svd(X, n_components):
    X = X.astype(np.float32, copy=False)
    mean = X.mean(axis=0, dtype=np.float32)
    Xc = X - mean
    # SVD PCA
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Embedding of [L,D] reduced to [K, D] where K is pca dimension
    components = Vt[:n_components].astype(np.float32) 
    return mean, components

def pca_transform(X, mean, components):
    return (X.astype(np.float32, copy=False) - mean) @ components.T

def pca_inverse(Z, mean, components):
    return Z @ components + mean

def normalized_layer_coords(L: int) -> np.ndarray:
    # map l=0..L-1 to t in [-1, 1]
    if L <= 1:
        return np.zeros((L,), dtype=np.float32)
    layers = np.arange(L, dtype=np.float32)
    t = 2.0 * (layers / float(L - 1)) - 1.0
    return t.astype(np.float32)

def fit_poly_coeffs_per_protein(pooled_LD, mean, components, degree=3):
    """
    pooled_LD: [L, D] 
    returns coeffs: [K, degree+1] ascending (c0..cd)
    """
    L, _ = pooled_LD.shape
    # Size = [L, K]    
    Z = pca_transform(pooled_LD, mean, components)  
    # For each layer L [L]
    t = normalized_layer_coords(L)                 

    K = Z.shape[1]
    coeffs = np.zeros((K, degree + 1), dtype=np.float32)
    for k in range(K):
        # cofficients = [cd..c0]
        c_desc = np.polyfit(t, Z[:, k], deg=degree)   
        # coff with each k [c0..cd]
        coeffs[k] = c_desc[::-1].astype(np.float32)   
    return coeffs

def eval_poly(coeffs_Kd1, t_scalar: float) -> np.ndarray:
    # coeffs in ascending order
    deg = coeffs_Kd1.shape[1] - 1
    # [deg+1]
    powers = np.array([t_scalar**j for j in range(deg + 1)], dtype=np.float32)  
    # [K] 
    return coeffs_Kd1 @ powers 

def reconstruct_from_coeffs(coeffs_Kd1, mean, components, layer_idx: int, L: int) -> np.ndarray:
    t = normalized_layer_coords(L)[layer_idx]
    # [1, K]
    z = eval_poly(coeffs_Kd1, float(t))[None, :]  
    x = pca_inverse(z, mean, components).reshape(-1)
    return x.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--n_pcs", type=int, default=64)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max_proteins_for_pca", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pooled = load_npz_embeddings(args.train_npz)
    pids = np.array(sorted(pooled.keys()))
    first = pooled[pids[0]]
    L, D = first.shape
    print(f"[load] proteins={len(pids)}  L={L} D={D}  dtype={first.dtype}")

    # subsample for PCA if requested
    rng = np.random.default_rng(args.seed)
    if args.max_proteins_for_pca and args.max_proteins_for_pca < len(pids):
        pids_for_pca = rng.choice(pids, size=args.max_proteins_for_pca, replace=False)
        print(f"[pca] subsample proteins={len(pids_for_pca)}")
    else:
        pids_for_pca = pids

    # stack all layers for PCA
    X = np.concatenate([pooled[pid].astype(np.float32) for pid in pids_for_pca], axis=0)
    print(f"[pca] X shape={X.shape}")

    K = min(args.n_pcs, D)
    mean, components = fit_pca_svd(X, n_components=K)
    print(f"[pca] fitted mean {mean.shape} components {components.shape}")

    # fit coeffs for each protein (train proteins)
    coeffs_all = np.zeros((len(pids), K, args.degree + 1), dtype=np.float32)
    for i, pid in enumerate(pids):
        coeffs_all[i] = fit_poly_coeffs_per_protein(pooled[pid], mean, components, degree=args.degree)
        if (i + 1) % 200 == 0:
            print(f"[poly] fitted {i+1}/{len(pids)}")

    # Reconstruction Error 
    sample = rng.choice(len(pids), size=min(10, len(pids)), replace=False)
    cos_sims, mses = [], []
    for idx in sample:
        pid = pids[idx]
        true_LD = pooled[pid].astype(np.float32)
        for layer in [0, L // 2, L - 1]:
            recon = reconstruct_from_coeffs(coeffs_all[idx], mean, components, layer, L)
            t = true_LD[layer]
            cos = float(np.dot(t, recon) / (np.linalg.norm(t) * np.linalg.norm(recon) + 1e-9))
            mse = float(np.mean((t - recon) ** 2))
            cos_sims.append(cos)
            mses.append(mse)
    print(f"[Reconstruction Error] cosine mean={np.mean(cos_sims):.4f}  mse mean={np.mean(mses):.6f}")

    # save: dense arrays 
    np.savez(
        args.out_npz,
        pids=pids,
        L=np.int32(L),
        D=np.int32(D),
        n_pcs=np.int32(K),
        degree=np.int32(args.degree),
        pca_mean=mean.astype(np.float32),
        pca_components=components.astype(np.float32),
        coeffs=coeffs_all.astype(np.float32),
    )
    print(f"[save] wrote {args.out_npz}")

if __name__ == "__main__":
    main()
