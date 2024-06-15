# Evaluating LLMs on M2QA
Use the `m2qa_llm_evaluation` conda environment from [environment.yml](environment.yml)

```bash
conda env create -f environment.yml
conda activate m2qa_llm_evaluation
```

We evaluated `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-0301` and `Llama 2-chat (13b)` on the M2QA dataset. For every model, we evaluated zero-shot and few-shot prompts. To reproduce our results:
1. Create predictions:
    - For OpenAI models: `python3 generate_openai_answers.py --model="<MODEL>" --prompt_name="<PROMPT>" --output_dir=<OUTPUR_DIR>`
      - Models that we used were: `gpt-3.5-turbo-0613` and `gpt-3.5-turbo-0301`
      - Prompts are listed in [prompts.py](prompts.py)
    - For Aya 23: `python3 generate_local_llm_answers.py --model_name="llama2" --output_dir=Results/aya23/` and before this set `PROMPT_FOR_LLAMA = True` in [prompts.py](prompts.py)
    - For Llama 2: `python3 generate_local_llm_answers.py --model_name="llama3" --output_dir=Results/llama2chat/` and before this set `PROMPT_FOR_LLAMA = True` in [prompts.py](prompts.py)
    - For Llama 3: `python3 generate_local_llm_answers.py --model_name="aya23"  --output_dir=Results/llama3instruct/` and before this set `PROMPT_FOR_LLAMA = False` in [prompts.py](prompts.py)
2. Evaluate predictions:
    - `python3 evaluate_llm_answers.py --input_dir=Results/Eval_on_M2QA_Validation_Data/gpt-35-turbo/gpt-35-turbo-0613/zero_shot_english/`

All model predictions have been made available in the [./Results](Results) directory, ensuring the reproducibility of our results and facilitating further analysis. To get the results, you have to unzip `Results/Eval_on_Train_data.zip` and `Results/Eval_on_M2QA_Validation_Data.zip`:

```bash
cd Results
unzip -P m2qa Eval_on_Train_data.zip # This zipped directory contains the LLM predictions on the M2QA training data
unzip -P m2qa Eval_on_M2QA_Validation_Data.zip # This contains the LLM predictions on the M2QA benchmark data (i.e. the results we reported in the paper)
```

## Case Study on Hard Questions (In Section 4.3)
In Section 4.3 (Results) of our paper, we made a case study on hard questions: To gain further insights into the performance of GPT-3.5, we manually inspected German questions for which all four GPT-3.5 setups achieved an F1 score lower than 25.

To reproduce these results, execute [analyse_llm_answer.py](analyse_llm_answer.py) and then randomly sample question-answer pairs from [Results/Analysis/low_f1_OR_missclassified_examples.json](Results/Analysis/low_f1_OR_missclassified_examples.json).

Our evaluation can be found here: [Results/Analysis/Case-study-hard-questions.ods](Results/Analysis/Case-study-hard-questions.ods). This table includes the questions, text passages, expected answers and the answers given by each model and - most importantly - our classification of why this given answer is wrong.

In this case study, we found that all GPT-3.5 models misclassified 20.9% of the German QA instances. We found that in 72% of the cases, the question and answer are correctly annotated in the data, but the model either makes erroneous predictions (58%) or generates a correct answer instead of extracting it (14%). We further identified issues with inconsistent annotations (22%, i.e. 4.6% of all German data), questions with multiple plausible answers (4%), and the evaluation metric (2%). 
