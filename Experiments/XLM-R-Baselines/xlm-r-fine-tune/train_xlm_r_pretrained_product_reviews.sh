#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=/mnt/beegfs/work/englaender/M2QA/m2qa/output/train_xlm_r/log_reviews_yelp.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --constraint="gpu_mem:40gb|gpu_mem:48gb|gpu_mem:80gb"

echo "laod cuda"
module load cuda

cd /storage/ukp/work/englaender/M2QA/m2qa/Experiments/xlm-r-fine-tune/

echo "start python script"
# the script uses early stoppng of 5
/storage/ukp/work/englaender/miniconda/envs/plain_hf_transformers/bin/python run_qa.py \
  --model_name_or_path /storage/ukp/work/englaender/M2QA/m2qa/Experiments/Trained_model/xlm-r-pretrained-reviews-yelp/ \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --max_steps 100000 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir /storage/ukp/work/englaender/M2QA/m2qa/Experiments/Trained_model/xlm-r-reviews-yelp-squad-512-64/ \
  --report_to wandb \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 1000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
