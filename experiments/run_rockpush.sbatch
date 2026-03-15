#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=wnn_rock
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/rockpush/rock_%j.log
#SBATCH --error=logs/rockpush/rock_%j.err

module load cuda

cd ~/scratch/WorldNN

mkdir -p logs/rockpush

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Rock-Push Experiment (obj-009) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

python experiments/rock_push.py --results-dir results --n-episodes 1000

echo ""
echo "=== Done: $(date) ==="
