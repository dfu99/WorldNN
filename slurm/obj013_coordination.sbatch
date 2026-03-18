#!/bin/bash
#SBATCH -J obj013-coord-quality
#SBATCH -A gts-yke8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/obj013_%j.out
#SBATCH -e logs/obj013_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda

# Activate environment
source ~/scratch/worldnn-env/bin/activate 2>/dev/null || {
    python3 -m venv ~/scratch/worldnn-env
    source ~/scratch/worldnn-env/bin/activate
    pip install torch numpy matplotlib
}

mkdir -p logs results

echo "=== obj-013: Coordination Quality C_i ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python experiments/coordination_quality.py

echo "End: $(date)"
echo "=== Done ==="
