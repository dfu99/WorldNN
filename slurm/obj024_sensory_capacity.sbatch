#!/bin/bash
#SBATCH -J obj024-sensory-capacity
#SBATCH -A gts-yke8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1
#SBATCH -C V100-16GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/obj024_%j.out
#SBATCH -e logs/obj024_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda
module load python/3.11.9

# Activate environment
source ~/scratch/worldnn-env-311/bin/activate 2>/dev/null || {
    python3 -m venv ~/scratch/worldnn-env-311
    source ~/scratch/worldnn-env-311/bin/activate
    pip install --upgrade pip
    pip install torch numpy matplotlib scipy
}

mkdir -p logs results

echo "=== obj-024: Sensory-Capacity Tradeoff ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Grid: 4 sensory_dims × 5 embed_dims × 5 seeds = 100 configs"
echo ""

python experiments/sensory_capacity_tradeoff.py

echo ""
echo "=== Done: $(date) ==="
