The MAD-X+Domain setup consists of 3 adapters:
1. Language Adapters
2. Domain Adapters
3. QA head adapter


First, activate the conda environment: `conda activate m2qa_experiments`  
Then train the adapters by starting the bash scripts:
1. The language adapters are those from MAD-X ([Pfeiffer et al., 2020](https://aclanthology.org/2020.emnlp-main.617/)), we do not need to train them again
2. Domain adapters are trained by executing the [DomainAdapter/start_train_domain_adapter.sh](DomainAdapter/start_train_domain_adapter.sh) bash script. Note: WANDB_KEY and DOMAIN ("wiki", "news", "creative_writing" or "product_reviews") must be set before execution.
3. The QA adapter head is trained by executing [QA_Head/start_train_squad_head_adapter.sh](QA_Head/start_train_squad_head_adapter.sh)
