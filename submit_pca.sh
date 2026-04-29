#!/bin/bash
#SBATCH --job-name=cubic_calib
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cubic_calib_%j.out
#SBATCH --error=logs/cubic_calib_%j.err

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate myevn
echo "--------------------------first"
python3 -u fit_cubic_surrogate.py \
  --train_npz emb_out/train_pooled_by_layer.npz \
  --out_npz cubic_model_train_K128_deg3.npz \
  --n_pcs 128 \
  --degree 3
echo "--------------------------second"
python3 fit_coeffs_given_train_pca.py \
  --train_model cubic_model_train_K128_deg3.npz \
  --val_npz emb_val_out/val_pooled_by_layer.npz \
  --out_npz cubic_coeffs_val_from_trainPCA_K128_deg3.npz

echo "----------------------------third"
python3 -u eval_val_real_vs_surrogate.py \
  --train_npz emb_out/train_pooled_by_layer.npz \
  --val_npz emb_val_out/val_pooled_by_layer.npz \
  --val_coeffs_npz cubic_coeffs_val_from_trainPCA_K128_deg3.npz \
  --train_pos train_pos.txt \
  --train_neg train_neg.txt \
  --val_pos val_pos.txt \
  --val_neg val_neg.txt \
  --mode scan_layers

# python3 fit_coeffs_given_train_pca.py \
#   --train_model cubic_model_train_K64_deg3.npz \
#   --val_npz emb_test_out/test_pooled_by_layer.npz \
#   --out_npz cubic_coeffs_test_from_trainPCA_K64_deg3.npz

# python3 eval_test_real_vs_surrogate.py \
#   --train_npz emb_out/train_pooled_by_layer.npz \
#   --test_npz emb_test_out/test_pooled_by_layer.npz \
#   --test_coeffs_npz cubic_coeffs_test_from_trainPCA_K64_deg3.npz \
#   --train_pos train_pos.txt \
#   --train_neg train_neg.txt \
#   --test_pos test_pos.txt \
#   --test_neg test_neg.txt \
#   --layer 12

