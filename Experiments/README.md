# Recreating Our Experiments
Below you find the instructions on how to reproduce the results of Sections 4 and 5 of the M2QA paper. For the results of Section "5.1 SQuAD Metric - Adaptation for Chinese" have a look at [M2QA_Metric/README.md](M2QA_Metric/README.md).


## XLM-R Baselines, MAD-X+Domain & MAD-X²
Use the `m2qa_experiments` conda environment from [environment.yml](environment.yml)

### Training
- **XLM-R-Base** and **XLM-R-Domain** are trained in the `XLM-R-Baselines/` directory: [XLM-R-Baselines/README.md](XLM-R-Baselines/README.md)
- **MAD-X+Domain**: Described in [mad-x-domain/README.md](mad-x-domain/README.md)
- **MAD-X²**: Described in [mad-x-2/README.md](mad-x-2/README.md)

We uploaded the trained adapters to Hugging Face: **TODO**

### Evaluation
The evaluation for our XLM-R Baselines, the MAD-X+Domain and MAD-X² setup are done via `evaluate_model.py`:
- Have a look at the available arguments by executing `python3 evaluate_model.py --help`
- E.g. to evaluate all models on SQuADv2, XQuAD and M2QA: `python3 evaluate_model.py --evaluate_xlm_r --evaluate_xlm_r_domain_adapted --evaluate_mad_x_domain --evaluate_squad --evaluate_xquad --evaluate_m2qa`


## LLMs
We evaluated `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-0301` and `Llama 2-chat (13b)` on the M2QA dataset. This is described in detail here: [LLM_evaluation/README.md](LLM_evaluation/README.md)

