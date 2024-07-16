#!/usr/bin/env bash
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=90G

python -u Train_SPC_original.py 7