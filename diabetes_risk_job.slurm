#!/bin/bash
#SBATCH --job-name=diabetes_ml
#SBATCH --output=diabetes_ml_%j.out
#SBATCH --error=diabetes_ml_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Week 11 HPC batch job for the diabetes-risk ML project.
# Confirm Python/module setup with the KENET/HPC training instructions.
# If required, add the exact partition/account directive supplied by the centre.

set -e

echo "Job started: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# Create this environment once on the login node:
#   python3 -m venv ~/venvs/diabetes-ml
#   source ~/venvs/diabetes-ml/bin/activate
#   pip install -r requirements.txt
#
# Then uncomment and edit this line to match your environment:
# source ~/venvs/diabetes-ml/bin/activate

# Alternatively load the Python module demonstrated during the session:
# module avail python
# module load python/3.x

python --version
python diabetes_risk_hpc.py

echo "Job finished: $(date)"
