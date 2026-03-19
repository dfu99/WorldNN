#!/bin/bash
#SBATCH -J obj014-expanded-ci
#SBATCH -A gts-yke8
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/obj014_%j.out
#SBATCH -e logs/obj014_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda
module load python/3.11.9

source ~/scratch/worldnn-env-311/bin/activate 2>/dev/null || {
    python3 -m venv ~/scratch/worldnn-env-311
    source ~/scratch/worldnn-env-311/bin/activate
    pip install --upgrade pip
    pip install torch numpy matplotlib
}

mkdir -p logs results

echo "=== obj-014: Expanded C_i Sweep ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python experiments/expanded_ci_sweep.py

echo "End: $(date)"
echo "=== Done ==="
