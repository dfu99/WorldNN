#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=wnn_pred
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/predictive/pred_%j.log
#SBATCH --error=logs/predictive/pred_%j.err

module load cuda

cd ~/scratch/WorldNN

# Create log directory
mkdir -p logs/predictive

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Predictive Processing Experiment (obj-007) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(python --version)"
echo ""

python experiments/predictive_processing.py --results-dir results

echo ""
echo "=== Done: $(date) ==="
