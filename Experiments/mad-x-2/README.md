The MAD-X² setup consists of 3 adapters:
1. MAD-X² language adapters
2. MAD-X² domain adapters
3. QA head adapter

First, activate the conda environment: `conda activate m2qa_experiments`  
Then train the adapters by starting the bash scripts:
1. Domain & Language Adapter:
    - Part of the german creative writing is the Corpus of German-Language Fiction ([Fischer et al. 2015](https://ds.ifi.uni-heidelberg.de/files/Team/jannik/publications/fischer-stroetgen_temporal-expressions-in-literary-corpora_dh2015_final_2015-03-01.pdf)), load it from here: https://figshare.com/articles/dataset/Corpus_of_German-Language_Fiction_txt_/4524680 and put it in the directory `Corpora/corpus-of-german-fiction-txt/`
    - Execute the [train_domain_language_adapter.py](train_domain_language_adapter.py) through the [start_training.sh](start_training.sh) bash file
3. The QA adapter head is trained by executing [QA_Head/start_train_squad_head_adapter.sh](QA_Head/start_train_squad_head_adapter.sh)
    - Note: In this `train_squad_head.py` script, you must edit the paths so that they point to the previously trained English language and Wikipedia domain adapters. 
