#!/bin/bash
#SBATCH -J obj015-ci-dynamics
#SBATCH -A gts-yke8
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/obj015_%j.out
#SBATCH -e logs/obj015_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda
module load python/3.11.9

source ~/scratch/worldnn-env-311/bin/activate

mkdir -p logs results

echo "=== obj-015: C_i Dynamics During Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python experiments/ci_dynamics.py

echo "End: $(date)"
echo "=== Done ==="
