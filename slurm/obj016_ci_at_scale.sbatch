#!/bin/bash
#SBATCH -J obj016-ci-at-scale
#SBATCH -A gts-yke8
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/obj016_%j.out
#SBATCH -e logs/obj016_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda
module load python/3.11.9

source ~/scratch/worldnn-env-311/bin/activate

mkdir -p logs results

echo "=== obj-016: C_i At Scale (280 configs, 7 seeds) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python experiments/ci_at_scale.py

echo "End: $(date)"
echo "=== Done ==="
