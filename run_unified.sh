#!/bin/bash
#SBATCH --job-name=unified_mlp
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -p gpu-a100,gpu-l40
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=75GB
#SBATCH --time=6-00:00:00

module load anaconda
conda activate my_env

python -u "30_exp_unified_8models.py"