#!/bin/bash
#SBATCH -J obj017-multirock-ci
#SBATCH -A gts-yke8
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/obj017_%j.out
#SBATCH -e logs/obj017_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

cd ~/scratch/WorldNN
module load cuda
module load python/3.11.9

source ~/scratch/worldnn-env-311/bin/activate

mkdir -p logs results

echo "=== obj-017: Multi-Rock C_i (8D state, 2nd task for ICLR) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python experiments/multirock_ci.py

echo "End: $(date)"
echo "=== Done ==="
