# Recreating Our Experiments
Below you find the instructions on how to reproduce the results of Sections 4 and 5 of the M2QA paper. For the results of Section "5.1 SQuAD Metric - Adaptation for Chinese" have a look at [M2QA_Metric/README.md](M2QA_Metric/README.md).


## XLM-R Baselines, MAD-X+Domain & MAD-X²
Use the `m2qa_experiments` conda environment from [environment.yml](environment.yml)

```bash
conda env create -f environment.yml
conda activate m2qa_experiments
```

### Training
- **XLM-R-Base** and **XLM-R-Domain** are trained in the `XLM-R-Baselines/` directory: [XLM-R-Baselines/README.md](XLM-R-Baselines/README.md)
- **MAD-X+Domain**: Described in [mad-x-domain/README.md](mad-x-domain/README.md)
- **MAD-X²**: Described in [mad-x-2/README.md](mad-x-2/README.md)

We uploaded the trained adapters to Hugging Face: [AdapterHub M2QA Adapter Collection](https://huggingface.co/collections/AdapterHub/m2qa-adapters-6660bf3752ba5ac56930c249)


### Evaluation
The evaluation for our XLM-R Baselines, the MAD-X+Domain and MAD-X² setup are done via `evaluate_model.py`:
- Have a look at the available arguments by executing `python3 evaluate_model.py --help`
- E.g. to evaluate all models on SQuADv2, XQuAD and M2QA: `python3 evaluate_model.py --evaluate_xlm_r --evaluate_xlm_r_domain_adapted --evaluate_mad_x_domain --evaluate_squad --evaluate_xquad --evaluate_m2qa`
- **Important:** If you want to evaluate other adapters than the ones trained by us and uploaded to Hugging Face, you have to exchange the paths to the adapters in the variable `PATHS` in `evaluate_model.py`.


## LLMs
We evaluated `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-0301`, `Llama 2-chat (13b)`, `Llama 3-instruct (8b)` and `Aya-23 (8b)` on the M2QA dataset. This is described in detail here: [LLM_evaluation/README.md](LLM_evaluation/README.md)

## MAD-X² / MAD-X+Domain Adapters Code Example
### MAD-X²
```python
from adapters import AutoAdapterModel
from adapters.composition import Stack

model = AutoAdapterModel.from_pretrained("xlm-roberta-base")

# 1. Load language adapter
language_adapter_name = model.load_adapter("AdapterHub/m2qa-xlm-roberta-base-mad-x-2-english") 

# 2. Load domain adapter
domain_adapter_name = model.load_adapter("AdapterHub/m2qa-xlm-roberta-base-mad-x-2-product-reviews")

# 3. Load QA head adapter
qa_adapter_name = model.load_adapter("AdapterHub/m2qa-xlm-roberta-base-mad-x-2-qa-head")

# 4. Activate them via the adapter stack
model.active_adapters = Stack(language_adapter_name, domain_adapter_name, qa_adapter_name)
```

### MAD-X+Domain
The code for loading and stacking the MAD-X+Domain adapters is the same except for the different adapter names:

```python
from adapters import AutoAdapterModel
from adapters.composition import Stack

model = AutoAdapterModel.from_pretrained("xlm-roberta-base")

# 1. Load language adapter
language_adapter_name = model.load_adapter("de/wiki@ukp") # MAD-X+Domain uses the MAD-X language adapter

# 2. Load domain adapter
domain_adapter_name = model.load_adapter("AdapterHub/m2qa-xlm-roberta-base-mad-x-domain-product-reviews")

# 3. Load QA head adapter
qa_adapter_name = model.load_adapter("AdapterHub/m2qa-xlm-roberta-base-mad-x-domain-qa-head")

# 4. Activate them via the adapter stack
model.active_adapters = Stack(language_adapter_name, domain_adapter_name, qa_adapter_name)
```
