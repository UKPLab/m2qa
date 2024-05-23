#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/M2QA/m2qa/output/train_domain_adapter/log.txt
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_mem:40gb|gpu_mem:48gb|gpu_mem:80gb"

echo "load cuda"
module load cuda

echo "run python"
/storage/ukp/work/englaender/miniconda/envs/m2qa_experiments/bin/python /storage/ukp/work/englaender/M2QA/m2qa/Experiments/DomainAdapter/train_madx_domain_adapter.py <WANDB_KEY> <DOMAIN>
