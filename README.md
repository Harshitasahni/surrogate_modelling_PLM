# Surrogate Modelling for ESM Embeddings

A polynomial surrogate model in PCA-reduced [ESM](https://github.com/facebookresearch/esm) embedding space for fast, layer-wise pairwise protein prediction. Instead of repeatedly running the full protein language model (PLM), we precompute per-layer mean-pooled embeddings once, project them into a low-dimensional PCA basis fit on the training set, and fit a cubic polynomial.

- Surrogate fit + evaluation is cheap polynomial algebra in a low-dimensional PCA space, so you can scan across layers and over many pairs without re-running ESM.
- This makes layer ablations, large-scale screening, and cross-validation tractable on modest hardware.

## Repository structure

```
surrogate_modelling_PLM/
├── run_embedding.py                      # Step 1: ESM embedding generation
├── fit_cubic_surrogate.py                # Step 2: PCA + cubic fit on train
├── fit_coeffs_given_train_pca.py         # Step 3: project val/test, refit coeffs
├── eval_val_real_vs_surrogate.py         # Step 4: validation evaluation
├── eval_test_real_vs_surrogate.py        # Step 5: test evaluation
├── slurm/
│   └── cubic_calib.sbatch                # Reference SLURM job for steps 2–4
├── train_pos.txt / train_neg.txt         # Training pairs
├── val_pos.txt   / val_neg.txt           # Validation pairs
├── test_pos.txt  / test_neg.txt          # Test pairs
└── README.md
```

## Installation

```bash
git clone https://github.com/Harshitasahni/surrogate_modelling_PLM.git
cd surrogate_modelling_PLM

conda create -n surrogate-plm python=3.10 -y
conda activate surrogate-plm

pip install torch numpy scikit-learn fair-esm
```

A GPU is recommended for Step 1; later steps run on CPU.

## Input formats

**FASTA** — one record per protein:

```
>P12345
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ...
```

**Pair files** — one pair per line, whitespace- or comma-separated, `#` for comments:

```
P12345  Q67890
P11111, Q22222
```

Pairs whose IDs are missing from the FASTA are dropped silently.

## Usage

### Step 1 — Generate ESM embeddings

Per-layer, mean-pooled, with sliding-window tiling for long sequences and deterministic sharding for parallel jobs.

```bash
python embed.py \
    --fasta data/sequences.fasta \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --model esm2_t33_650M_UR50D \
    --layers 0-33 \
    --batch_size 4 --fp32 --device cuda \
    --num_shards 2 --shard_id 0 \
    --chunk_size 200 \
    --max_len 1024 --stride 512 \
    --out_root emb_out
```

Writes `emb_out/shard_{id}/chunk_{NNNNN}.npz`. Merge shards into a single `train_pooled_by_layer.npz` (and analogous files for val/test) before Step 2.

### Step 2 — Fit cubic surrogate on train

```bash
python fit_cubic_surrogate.py \
    --train_npz emb_out/train_pooled_by_layer.npz \
    --out_npz   cubic_model_train_K128_deg3.npz \
    --n_pcs 128 --degree 3
```

### Step 3 — Project validation embeddings into the train PCA basis

```bash
python fit_coeffs_given_train_pca.py \
    --train_model cubic_model_train_K128_deg3.npz \
    --val_npz     emb_val_out/val_pooled_by_layer.npz \
    --out_npz     cubic_coeffs_val_from_trainPCA_K128_deg3.npz
```

### Step 4 — Evaluate real vs. surrogate on validation

```bash
python eval_val_real_vs_surrogate.py \
    --train_npz emb_out/train_pooled_by_layer.npz \
    --val_npz   emb_val_out/val_pooled_by_layer.npz \
    --val_coeffs_npz cubic_coeffs_val_from_trainPCA_K128_deg3.npz \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --val_pos   val_pos.txt   --val_neg   val_neg.txt \
    --mode scan_layers
```

`--mode scan_layers` reports surrogate-vs-real agreement at every layer.

### Step 5 — (Optional) test set

Same shape as Steps 3–4 with test embeddings and pair files (see commented block in `slurm/cubic_calib.sbatch`).

## Running on SLURM

Steps 2–4 are wrapped in `slurm/cubic_calib.sbatch`:

```bash
sbatch slurm/cubic_calib.sbatch
```

Defaults: 1 node, 16 CPUs, 32 GB RAM, 1 hour. Update `--partition`, `--time`, and the conda env name for your cluster.

## Key design choices

- **Mean pooling, not CLS.** Per-residue representations are mean-pooled with BOS / EOS / PAD tokens excluded.
- **Length-weighted window averaging.** Sequences longer than `max_len` are split into overlapping windows; window embeddings are averaged weighted by window length.
- **Train-only PCA.** PCA is fit on training embeddings; val and test are projected through the same basis to prevent leakage.
- **Per-layer surrogates.** A separate cubic is fit at each layer, so `scan_layers` can identify the best layer for the task.
- **OOM-resilient.** Batched short sequences fall back to per-protein retries on CUDA OOM; sequences that still don't fit are logged and skipped.

## Configuration

| Hyperparameter | Default | Where |
|---|---|---|
| ESM model | `esm2_t33_650M_UR50D` | `embed.py --model` |
| Window length / stride | 1024 / 512 | `embed.py --max_len`, `--stride` |
| PCA components `K` | 128 | `fit_cubic_surrogate.py --n_pcs` |
| Polynomial degree | 3 | `fit_cubic_surrogate.py --degree` |
| Embedding dtype | fp32 (with `--fp32`) | `embed.py` |

## Citation



## Contact
Maintainer: Harshita Sahni - hsahni@unm.edu
Trilce Estrada - trilce@unm.edu

