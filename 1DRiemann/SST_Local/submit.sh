#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --output=test_output.%j.log
#SBATCH --error=test_error.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=10:02:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=mkvkss


module load gcc-12.2.0
module load cuda-12.4
module load mpich-4.3-cuda
module load hdf5-1.14-parallel-mpich-4.3-cuda
module load openucx-1.17-cuda

source ../../../../python/myenv/bin/activate
python pinn.py
