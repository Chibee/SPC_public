#!/usr/bin/env bash
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --array=1-1000

python -u Calc_dist.py $SLURM_ARRAY_TASK_ID 5

