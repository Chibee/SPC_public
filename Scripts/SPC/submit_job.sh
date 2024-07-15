#!/usr/bin/env bash
#-------slurm option--------#
#SBATCH --partition=mpc
#SBATCH --account=RB230049
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --mem=90G

python -u Train_SPC_original.py 7