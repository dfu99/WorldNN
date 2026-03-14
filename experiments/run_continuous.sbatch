#!/bin/bash
#SBATCH -A gts-yke8
#SBATCH --job-name=wnn_cont
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu
#SBATCH --output=logs/continuous/cont_%j.log
#SBATCH --error=logs/continuous/cont_%j.err

module load cuda

cd ~/scratch/WorldNN

mkdir -p logs/continuous

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Continuous State Experiment (obj-008) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

python experiments/continuous_state.py --results-dir results

echo ""
echo "=== Done: $(date) ==="
