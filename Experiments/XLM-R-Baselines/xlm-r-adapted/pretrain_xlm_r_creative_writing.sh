#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/M2QA/m2qa/output/train_domain_adapter/log_books.txt
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_mem:40gb|gpu_mem:48gb|gpu_mem:80gb"

echo "laod cuda"
module load cuda

cd /storage/ukp/work/englaender/M2QA/m2qa/Experiments/xlm-r-adapted/

echo "start python script"
/storage/ukp/work/englaender/miniconda/envs/plain_hf_transformers/bin/python run_mlm.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name bookcorpus \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --mlm_probability 0.4 \
  --max_steps 100000 \
  --max_seq_length 512 \
  --output_dir /storage/ukp/work/englaender/M2QA/m2qa/Experiments/Trained_model/xlm-r-pretrained-books-512/ \
  --report_to wandb \
  --save_steps 5000 \
  --eval_steps 5000 \
  --logging_steps 1000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --overwrite_output_dir \
  --seed 1000
