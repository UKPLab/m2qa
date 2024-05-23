#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/M2QA/m2qa/output/train_squad_head/mad-x-domain-qa-head.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_mem:40gb|gpu_mem:48gb|gpu_mem:80gb"

echo "load cuda"
module load cuda

WANDB_API_KEY="<YOUR_KEY>"

echo "run python"
/storage/ukp/work/englaender/miniconda/envs/m2qa_experiments/bin/python /storage/ukp/work/englaender/M2QA/m2qa/Experiments/mad-x-domain/QA_Head/train_squad_head.py $WANDB_API_KEY

