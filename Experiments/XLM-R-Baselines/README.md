First, activate the conda environment: `conda activate m2qa_experiments`  
Then train the XLM-R-Base & XLM-R-Domain models by starting the bash scripts:

- **XLM-R Base**:
    - Execute [xlm-r-fine-tune/train_full_xlm_r_on_squad.sh](xlm-r-fine-tune/train_full_xlm_r_on_squad.sh)
- **XLM-R Domain Adapted**:
    1. Pretrain the XLM-R model on the target domain: Execute the bash files in the [xlm-r-adapted](xlm-r-adapted) directory
    2. Fine-tune each pretrained XLM-R model on SQuAD v2. The bash scripts to do this are in the Execute [xlm-r-fine-tune](xlm-r-fine-tune) directory
    - Example to train the XLM-R-Domain model for the creative writing domain:
        1. Execute [xlm-r-adapted/pretrain_xlm_r_creative_writing.sh](xlm-r-adapted/pretrain_xlm_r_creative_writing.sh)
        2. Execute [xlm-r-fine-tune/train_xlm_r_pretrained_creative_writing.sh](xlm-r-fine-tune/train_xlm_r_pretrained_creative_writing.sh)
