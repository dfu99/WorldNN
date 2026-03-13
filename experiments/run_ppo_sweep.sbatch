#!/bin/bash
#SBATCH -J worldnn-ppo-sweep
#SBATCH -A gts-yke8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

module load cuda
module load anaconda3

cd "$SLURM_SUBMIT_DIR"

# Create venv if it doesn't exist, install deps
VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA and project deps
pip install --quiet torch numpy matplotlib scipy pyyaml tqdm 2>&1 | tail -5
pip install -e . --quiet 2>&1 | tail -3

echo "Python: $(python --version), Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run PPO sweep on GPU (auto-detects CUDA)
python experiments/perturbation_study_ppo.py --results-dir results
