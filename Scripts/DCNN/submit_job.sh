#!/usr/bin/env bash
#-------slurm option--------#
#SBATCH --partition=mpc
#SBATCH --account=RB230049
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --mem=8G
#SBATCH --array=1-222

python -u Calc_dist.py $SLURM_ARRAY_TASK_ID 5

