#!/bin/bash
#SBATCH --job-name=ppi_prott5
#SBATCH --partition=l40s          # <-- cluster-specific: change to your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- environment (cluster-specific: edit these two lines) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate surrogate-plm          # create with: conda env create -f environment.yml

'''
#For ProtT5

# ---- Stage 1: generate ProtT5 embeddings (skip if .npz already present) ----
# python run_embeddings.py \
#   --fasta data/ppi.fasta --all_proteins \
#   --model prott5 --layers 0-24 \
#   --batch_size 32 --fp32 --device cuda \
#   --num_shards 1 --shard_id 0 --chunk_size 500 \
#   --out_root ppi_pooled_prott5

# ---- Stage 2: fit PCA + cubic surrogate on TRAIN ----
echo "=== Stage 2: fit surrogate (train) ==="
python3 fit_cubic_surrogate.py \
  --train_npz ppi_pooled_prott5_train.npz \
  --out_npz   prott5_cubic_model_train_K64_deg3.npz \
  --n_pcs 64 --degree 3

# ---- Stage 3: fit per-protein coefficients for TEST using frozen train PCA ----
echo "=== Stage 3: fit test coefficients ==="
python3 fit_coeffs_given_train_pca.py \
  --train_model prott5_cubic_model_train_K64_deg3.npz \
  --val_npz     ppi_pooled_prott5_test.npz \
  --out_npz     prott5_cubic_coeffs_test_K64_deg3.npz

# ---- Stage 4: evaluate original vs. surrogate at layer 24 ----
echo "=== Stage 4: eval original vs surrogate ==="
python3 eval_test_real_vs_surrogate.py \
  --train_npz ppi_pooled_prott5_train.npz \
  --test_npz  ppi_pooled_prott5_test.npz \
  --test_coeffs_npz prott5_cubic_coeffs_test_K64_deg3.npz \
  --train_pos train_pos.txt --train_neg train_neg.txt \
  --test_pos  test_pos.txt  --test_neg  test_neg.txt \
  --layer 24
  
  '''

# For PPI 

MODEL=esm2_t12_35M_UR50D        # ESM-35M; use esm2_t36_3B_UR50D for ESM-3B
PREFIX=ppi_pooled_35M           # use ppi_pooled_3B for ESM-3B
LAYER=12                        # 12 for ESM-35M, 32 for ESM-3B

# ---- Stage 1: generate embeddings (skip if present) ----
# python run_embeddings.py \
#   --fasta data/ppi.fasta --all_proteins \
#   --model $MODEL --layers 0-12 \
#   --batch_size 32 --fp32 --device cuda \
#   --num_shards 1 --shard_id 0 --chunk_size 500 \
#   --out_root $PREFIX

echo "=== Stage 2: fit surrogate (train) ==="
python3 fit_cubic_surrogate.py \
  --train_npz ${PREFIX}_train.npz \
  --out_npz   ${PREFIX}_cubic_model_K64_deg3.npz \
  --n_pcs 64 --degree 3

echo "=== Stage 3: fit test coefficients ==="
python3 fit_coeffs_given_train_pca.py \
  --train_model ${PREFIX}_cubic_model_K64_deg3.npz \
  --val_npz     ${PREFIX}_test.npz \
  --out_npz     ${PREFIX}_cubic_coeffs_test_K64_deg3.npz

# ---- Optional Stage 3b: layer scan on validation (how layer was chosen) ----
echo "=== Stage 3b: validation layer scan ==="
python3 fit_coeffs_given_train_pca.py \
  --train_model ${PREFIX}_cubic_model_K64_deg3.npz \
  --val_npz     ${PREFIX}_val.npz \
  --out_npz     ${PREFIX}_cubic_coeffs_val_K64_deg3.npz
  
python3 select_optimal_layer_using_validation.py \
  --train_npz ${PREFIX}_train.npz --val_npz ${PREFIX}_val.npz \
  --val_coeffs_npz ${PREFIX}_cubic_coeffs_val_K64_deg3.npz \
  --train_pos train_pos.txt --train_neg train_neg.txt \
  --val_pos val_pos.txt --val_neg val_neg.txt \
  --mode scan_layers

echo "=== Stage 4: eval original vs surrogate at layer $LAYER ==="
python3 eval_test_real_vs_surrogate.py \
  --train_npz ${PREFIX}_train.npz --test_npz ${PREFIX}_test.npz \
  --test_coeffs_npz ${PREFIX}_cubic_coeffs_test_K64_deg3.npz \
  --train_pos train_pos.txt --train_neg train_neg.txt \
  --test_pos  test_pos.txt  --test_neg  test_neg.txt \
  --layer $LAYER


  '''

  PREFIX=deeploc_pooled_35M       # use deeploc_pooled_3B for ESM-3B

# ---- Stage 1: generate embeddings (skip if present) ----
# python run_embeddings.py \
#   --fasta deeploc/deeploc_data.fasta --all_proteins \
#   --model esm2_t12_35M_UR50D --layers 0-12 \
#   --batch_size 32 --fp32 --device cuda \
#   --num_shards 1 --shard_id 0 --chunk_size 500 \
#   --out_root $PREFIX

echo "=== Stage 2: fit surrogate (train), K=128 ==="
python3 fit_cubic_surrogate.py \
  --train_npz ${PREFIX}_train.npz \
  --out_npz   surrogate_${PREFIX}_K128_deg3.npz \
  --n_pcs 128 --degree 3

echo "=== Stage 3: fit test coefficients ==="
python3 fit_coeffs_given_train_pca.py \
  --train_model surrogate_${PREFIX}_K128_deg3.npz \
  --val_npz     ${PREFIX}_test.npz \
  --out_npz     surrogate_${PREFIX}_test_coeffs_K128_deg3.npz

echo "=== Stage 4: reconstruct surrogate embeddings ==="
python3 reconstruct_surrogate.py \
  --coeffs_files surrogate_${PREFIX}_K128_deg3.npz \
                 surrogate_${PREFIX}_test_coeffs_K128_deg3.npz \
  --out_npz      ${PREFIX}_surrogate_K128_deg3.npz

echo "=== Stage 5: evaluate localization ==="
python3 eval_localization.py \
  --embeddings ${PREFIX}_surrogate_K128_deg3.npz \
  --labels deeploc/labels.csv \
  --out results_surrogate_${PREFIX}_K128_deg3.csv

'''

  
