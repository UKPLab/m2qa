import os
import sys

import wandb
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PfeifferConfig,
    TrainingArguments,
    XLMRobertaAdapterModel,
    set_seed,
)
from transformers.adapters.composition import Stack

# Command line arguments
# 1. Wandb API key
wandb.login(key=sys.argv[1])

# 2. Domain to train adapteron
ADAPTER_NAME = sys.argv[2]  # "wiki", "news", ...

#####################
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
TRAINING_STEPS = 100000
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
SEED = 123
DATASET_SHUFFLE_SEED = 42
NUM_WORKERS = 8

SAVE_ADAPTER_PATH = f"/storage/ukp/work/englaender/M2QA/m2qa/Experiments/Trained_adapters/{ADAPTER_NAME}/"
OUTPUT_DIR = f"/storage/ukp/work/englaender/M2QA/m2qa/Experiments/Train_output/{ADAPTER_NAME}"
SAVE_PREPROCESSED_DATASET_PATH = "/storage/ukp/work/englaender/M2QA/m2qa/Experiments/Preprocessed_datasets/"

CHUNK_SIZE = 512  # is xlm-r tokenizer model_max_length

# Lists of datasets to train on. Each entry is a dict containg a function to load the dataset, a tokenize function and language
DATASETS_TO_TRAIN_ON = {
    "wiki": [
        {
            "language": "english",
            "name": "wikipedia",
            "load_dataset": lambda: load_dataset("wikipedia", "20220301.en"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset),
            "after_shuffle_function": (
                lambda dataset: dataset["train"].select(range(2000000)).train_test_split(test_size=0.15)
            ),
        }
    ],
    "news": [
        {
            "name": "cnn_dailymail",
            "language": "english",
            "load_dataset": lambda: load_dataset("cnn_dailymail", "3.0.0"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="article"),
        }
    ],
    "creative_writing": [
        {
            "language": "english",
            "name": "bookcorpus",
            "load_dataset": lambda: load_dataset("bookcorpus"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset),
        }
    ],
    "product_reviews": [
        {
            "language": "english",
            "name": "amazon",
            "load_dataset": lambda: load_dataset(
                "/ukp-storage-1/englaender/.cache/huggingface/datasets/amazon_reviews_multi/en/1.0.0/"
            ),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset),
        }
    ],
}

LANGUAGE_ADAPTER = {
    "english": {
        "adapter_name": "en",
        "adapter_path": "en/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
    "german": {
        "adapter_name": "de",
        "adapter_path": "de/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
    "chinese": {
        "adapter_name": "zh",
        "adapter_path": "zh/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
    "arabic": {
        "adapter_name": "ar",
        "adapter_path": "ar/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
    "hindi": {
        "adapter_name": "hi",
        "adapter_path": "hi/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
    "turkish": {
        "adapter_name": "tr",
        "adapter_path": "tr/wiki@ukp",
        "adapter_config": AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2),
    },
}


def standard_preprocess_dataset(dataset: DatasetDict, text_field="text"):
    # dataset = dataset.flatten()
    return dataset.map(
        lambda examples: tokenizer([" ".join(x) for x in examples[text_field]], return_special_tokens_mask=True),
        batched=True,
        num_proc=NUM_WORKERS,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on every text in dataset",
    )


# This function is from Hugging Face
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE
    # Split by chunks of max_len
    result = {
        k: [t[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def load_and_preprocess_dataset(dataset_information: dict):
    language = dataset_information["language"]
    name = dataset_information["name"]

    # Check if dataset is already preprocessed
    if os.path.exists(f"{SAVE_PREPROCESSED_DATASET_PATH}/{name}_{language}"):
        print(f"Loading preprocessed dataset {dataset_information['name']}")
        dataset = load_from_disk(f"{SAVE_PREPROCESSED_DATASET_PATH}/{name}_{language}")

    else:
        print(f"Loading and preprocessing dataset {dataset_information['name']}")
        dataset = dataset_information["load_dataset"]()
        dataset = dataset.shuffle(DATASET_SHUFFLE_SEED)
        dataset = dataset_information["tokenize_fn"](dataset)
        dataset = dataset.map(group_texts, batched=True, num_proc=NUM_WORKERS)
        dataset.save_to_disk(f"{SAVE_PREPROCESSED_DATASET_PATH}/{name}_{language}")

    # Limit the train dataset to 100,000 * 64 (steps * batch size)
    if dataset["train"].num_rows > 6400000:
        dataset["train"] = dataset["train"].select(range(6400000))

    print(f"dataset: {dataset}")

    return dataset, language


def main():
    # Set seed before initializing model.
    print("Setting seed")
    set_seed(SEED)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(SAVE_ADAPTER_PATH):
        os.makedirs(SAVE_ADAPTER_PATH)

    # 0. Prepare tokenozer + data collator
    print("Preparing tokenizer and data collator")
    tokenizer.pad_token = tokenizer.eos_token
    lm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.4
    )  # XLM-R has a comparable size to bert-large and thus 0.4 is probably the best mlm_probability: https://arxiv.org/pdf/2202.08005.pdf

    # 1. Load model
    print("Loading model")
    model = XLMRobertaAdapterModel.from_pretrained(MODEL_NAME)

    # 2. Add MLM head and domain adapter to model
    print("Adding MLM head and domain adapter to model")
    model.add_masked_lm_head(ADAPTER_NAME)
    config = PfeifferConfig(reduction_factor=2)
    model.add_adapter(ADAPTER_NAME, config=config)

    # 3. Train the domain adapter for every dataset in DATASETS_TO_TRAIN_ON
    # Using multiple datasets in different languages could result in a better domain adapter
    for dataset_information in DATASETS_TO_TRAIN_ON[ADAPTER_NAME]:
        # 3.1 load dataset + preprocess
        dataset, language = load_and_preprocess_dataset(dataset_information)

        # 3.2 Add language adapter
        model.load_adapter(
            LANGUAGE_ADAPTER[language]["adapter_path"],
            config=LANGUAGE_ADAPTER[language]["adapter_config"],
        )

        # 3.3 Train model
        model.train_adapter([ADAPTER_NAME])
        model.active_adapters = Stack(LANGUAGE_ADAPTER[language]["adapter_name"], ADAPTER_NAME)

        print(f"For dataset {dataset_information['name']}: Training this" f" model:\n{model.adapter_summary()}")
        training_args = TrainingArguments(
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=1e-4,  # for higher learning_rate 2e-4
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            max_steps=TRAINING_STEPS,
            report_to="wandb",
            dataloader_num_workers=NUM_WORKERS,
            save_steps=5000,
            logging_steps=1000,
            warmup_steps=1000,
            lr_scheduler_type="linear",  # cosine
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=tokenizer,
            data_collator=lm_data_collator,
        )

        trainer.train()

    # 4. Save adapter
    model.save_adapter(SAVE_ADAPTER_PATH, ADAPTER_NAME, with_head=False)


if __name__ == "__main__":
    main()
