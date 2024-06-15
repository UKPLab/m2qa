#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/M2QA/m2qa/Experiments/storage/output/generate/llama.log
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_mem:40gb|gpu_mem:48gb|gpu_mem:80gb"

echo "load cuda"
module load cuda

TOKEN="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" # This token has been disabled, exchange it with your own token

echo "run python"
cd /storage/ukp/work/englaender/M2QA/m2qa/Experiments/LLM_evaluation/
/storage/ukp/work/englaender/miniconda/envs/m2qa_llm_evaluation/bin/python generate_local_llm_answers.py --model_name="llama3" --output_dir="/storage/ukp/work/englaender/M2QA/m2qa/Experiments/LLM_evaluation/llama2" --huggingface_token $TOKEN --limit 10
