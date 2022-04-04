#!/bin/sh
#SBATCH --job-name=name-of-job # Name of the job, can be useful for identifying your job later
#SBATCH --cpus-per-task=4      # Maximum amount of CPU cores (per MPI process)
#SBATCH --gres=gpu:v100:1      # GPU Resources
#SBATCH --mem=16G              # Maximum amount of memory (RAM)
#SBATCH --time=0-10:00         # Time limit (DD-HH:MM)
#SBATCH --nice=100             # Allow other priority jobs to go first

echo "CUDA_VISABLE_DEVICES=$CUDA_VISABLE_DEVICES"

# Set path
cd /scratch/lgdejong/projects/RecSeg/

# Set up enivronment modules and conda environment
conda activate thesis

# run the thing
srun python 