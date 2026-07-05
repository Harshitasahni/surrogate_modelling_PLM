# Surrogate Modelling for PLM Embeddings

A polynomial surrogate model in PCA-reduced protein language model (PLM) embedding
space for fast, layer-wise protein prediction. Instead of repeatedly running the full
PLM, we precompute per-layer mean-pooled embeddings once, project them into a
low-dimensional PCA basis fit on the training set, and fit a per-protein cubic
polynomial across layers. Any layer can then be reconstructed on demand from a few
stored coefficients.

- Surrogate fit and reconstruction are cheap polynomial algebra in a low-dimensional
  PCA space, so you can scan across layers and over many proteins without re-running
  the PLM.
- This makes layer ablations, large-scale screening, and cross-validation tractable on
  modest hardware.

Accompanies the paper *Polynomial Trajectory Compression for Protein Language Model
Embeddings* (see [Citation](#citation)). The pipeline is evaluated on ESM2-35M,
ESM2-3B, and ProtT5, across protein–protein interaction (PPI) and subcellular
localization tasks.

## Repository structure


## Installation

```bash
git clone https://github.com/Harshitasahni/surrogate_modelling_PLM.git
cd surrogate_modelling_PLM

conda env create -f environment.yml
conda activate surrogate-plm
```

## Input formats

**FASTA** — one record per protein:

P12345
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ...

**Pair files** — one pair per line, whitespace- or comma-separated, `#` for comments:
P12345  Q67890
P11111, Q22222

## Usage

### Step 1a — Generate ESM embeddings

Per-layer, mean-pooled, with sliding-window tiling for long sequences and deterministic sharding for parallel jobs.

```bash
python run_embeddings.py \
    --fasta data/sequences.fasta \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --model esm2_t12_35M_UR50D \
    --layers 0-12 \
    --batch_size 4 --fp32 --device cuda \
    --num_shards 2 --shard_id 0 \
    --chunk_size 200 \
    --max_len 1024 --stride 512 \
    --out_root emb_out
```
# Subcellular localization: embed all proteins (no pair filtering)
python run_embeddings.py \
    --fasta deeploc/deeploc_data.fasta \
    --all_proteins \
    --model esm2_t12_35M_UR50D --layers 0-12 \
    --batch_size 32 --fp32 --device cuda \
    --num_shards 1 --shard_id 0 \
    --chunk_size 500 \
    --out_root deeploc_emb_35M
	
The paper uses three models:
`esm2_t12_35M_UR50D` (layers 0-12), `esm2_t36_3B_UR50D` (layers 0-36), and ProtT5
(Step 1b). Writes `emb_out/shard_{id}/chunk_{NNNNN}.npz`; merge shards into `train_pooled_by_layer.npz` (and val/test analogues) before Step 2.

### Step 1b — Generate ProtT5 embeddings

Same mean-pooling and windowing, using the ProtT5 loader instead of `fair-esm`.
```bash
python run_embeddings_prott5.py \
    --fasta data/sequences.fasta \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --model Rostlab/prot_t5_xl_half_uniref50-enc \
    --layers 0-24 \
    --batch_size 1 --device cuda \
    --num_shards 1 --shard_id 0 \
    --chunk_size 200 \
    --max_len 1024 --stride 512 \
    --out_root emb_out_prott5
```


### Step 2 — Fit cubic surrogate on train

```bash
python fit_cubic_surrogate.py \
    --train_npz emb_out/train_pooled_by_layer.npz \
    --out_npz   cubic_model_train_K128_deg3.npz \
    --n_pcs 128 --degree 3
```
Use `--n_pcs 64` for PPI and `--n_pcs 128` for subcellular localization, matching the paper's configurations.


### Step 3 — Project val/test embeddings into the train PCA basis

```bash
python fit_coeffs_given_train_pca.py \
    --train_model cubic_model_train_K128_deg3.npz \
    --val_npz     emb_val_out/val_pooled_by_layer.npz \
    --out_npz     cubic_coeffs_val_from_trainPCA_K128_deg3.npz
```
PCA is fit on training embeddings only; val and test are projected through the same basis to prevent leakage.

### Step 4 — Validation layer scan

```bash
python select_optimal_layer_using_validation.py \
    --train_npz emb_out/train_pooled_by_layer.npz \
    --val_npz   emb_val_out/val_pooled_by_layer.npz \
    --val_coeffs_npz cubic_coeffs_val_from_trainPCA_K128_deg3.npz \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --val_pos   val_pos.txt   --val_neg   val_neg.txt \
    --mode scan_layers
```
`--mode scan_layers` reports surrogate-vs-real agreement at every layer, used to select the layer reported in the paper (12 for ESM2-35M, 32 for ESM2-3B, 24 for ProtT5).

### Step 5 — Test evaluation

**PPI** (real vs. surrogate at the selected layer):

```bash
python eval_test_real_vs_surrogate.py \
    --train_npz emb_out/train_pooled_by_layer.npz \
    --test_npz  emb_test_out/test_pooled_by_layer.npz \
    --test_coeffs_npz cubic_coeffs_test_from_trainPCA_K64_deg3.npz \
    --train_pos train_pos.txt --train_neg train_neg.txt \
    --test_pos  test_pos.txt  --test_neg  test_neg.txt \
    --layer 12
```

**Subcellular localization** (reconstruct surrogate embeddings, then evaluate):

```bash
python reconstruct_surrogate.py \
    --coeffs_files cubic_model_train_K128_deg3.npz \
                   cubic_coeffs_test_from_trainPCA_K128_deg3.npz \
    --out_npz      deeploc_surrogate_K128_deg3.npz

python eval_localization.py \
    --embeddings deeploc_surrogate_K128_deg3.npz \
    --labels     deeploc/labels.csv \
    --out        results_surrogate_K128_deg3.csv
```

## Running on SLURM

Embedding generation and evaluation must run on a compute node (not the login node).
Reference jobs are provided.

## Data

The PPI pairs are from Bernett et al. (2024); the subcellular localization data is the
DeepLoc 1.0 benchmark (Almagro Armenteros et al., 2017). Please obtain these datasets
from their original sources and regenerate embeddings. Place sequences as FASTA and pair lists (`*_pos.txt` / `*_neg.txt`) in the working
directory, then generate embeddings with Step 1.

## Citation

```bibtex

@article {Sahni2026.06.05.730461,
	author = {Sahni, Harshita and Chen, Xin and Estrada, Trilce},
	title = {Polynomial Trajectory Compression for Protein Language Model Embeddings},
	elocation-id = {2026.06.05.730461},
	year = {2026},
	doi = {10.64898/2026.06.05.730461},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/06/07/2026.06.05.730461},
	eprint = {https://www.biorxiv.org/content/early/2026/06/07/2026.06.05.730461.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Maintainer: Harshita Sahni — hsahni@unm.edu
Trilce Estrada — trilce@unm.edu
