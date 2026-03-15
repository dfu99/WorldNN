#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=wnn_oracle
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/oracle/oracle_%j.log
#SBATCH --error=logs/oracle/oracle_%j.err

module load cuda

cd ~/scratch/WorldNN
source .venv/bin/activate

echo "=== Oracle GPU Baseline (obj-009-expanded) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

python experiments/oracle_gpu.py
