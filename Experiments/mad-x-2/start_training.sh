#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/thesis/output/train_pure_adapters/log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --constraint="gpu_mem:80gb"

echo "load cuda"
module load cuda

echo "run python"
cd /storage/ukp/work/englaender/M2QA/m2qa/Experiments/mad-x-2
/storage/ukp/work/englaender/miniconda/envs/m2qa_experiments/bin/python train_domain_language_adapter.py
