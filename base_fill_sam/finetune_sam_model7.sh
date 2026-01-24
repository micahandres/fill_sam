#!/bin/bash

# ===== SLURM directives =====

#SBATCH --partition=gpu                                 # Use the GPU partition (enables GPU access)
#SBATCH --gres=gpu:1                                    # Request 1 GPU (e.g., RTX 3090 or A100)
#SBATCH --cpus-per-task=4                               # Request 4 CPU cores for data loading or parallel CPU work
#SBATCH --mem=32G                                       # Request 32 GB of RAM (system memory, not GPU VRAM)
#SBATCH --time=24:00:00                                 # Max runtime: 24 hours and 00 minutes (your account allows 48 max)
#SBATCH --job-name=model7_finetune                      # Name your job 'train_sam' for job queue tracking
#SBATCH --output=../logs/model7_logs/model7_%j.out      # Save standard output (stdout) to this file
#SBATCH --error=../logs/model7_logs/model7_%j.err       # Save standard error (stderr) to this file
#SBATCH --mail-type=END,FAIL                            # Get an email when the job ends or fails
#SBATCH --mail-user=mikaela_san_andres@brown.edu        # Email to notify for job events

# ===== Activate Conda =====
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# ===== setup environment =====
conda activate running_sam_conda_trial5

# ===== Move to job submission directory =====
cd "$SLURM_SUBMIT_DIR"

# ===== Run your script =====
python ../model7_finetune.py