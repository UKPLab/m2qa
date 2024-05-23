from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from hyper_parameter import (
    BATCH_SIZE,
    CHUNK_SIZE,
    DATASET_SHUFFLE_SEED,
    DOMAINS,
    LANGUAGES,
    MODEL_NAME,
    PREPROCESSED_DATASET_PATH,
    STEPS_PER_LANGUAGE_DOMAIN,
)
from transformers import AutoTokenizer

from pathlib import Path

# Need to load tokenizer outside of the main function else the batched map probably doesn't work (https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/17)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# This dict contains datasets for the language - domain combinations we want to train on
DATASETS = {
    "english": {
        "wikipedia": {
            "name": "wikipedia",
            # Use dataset processed by Hugging Face to save time
            "load_dataset_fn": lambda: load_dataset("wikipedia", "20220301.en"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset),
            "cut_off": 1_000_000,
        },
        "product_reviews": {
            "name": "amazon_reviews_multi",
            # The Multilingual Amazon Reviews Corpus is no longer available on Hugging Face Datasets, we need to use a downloaded version
            "load_dataset_fn": lambda: load_dataset(
                "/ukp-storage-1/englaender/.cache/huggingface/datasets/amazon_reviews_multi/en/1.0.0/"
            ),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="review_body"),
        },
        "news": {
            "name": "cnn_dailymail",
            "load_dataset_fn": lambda: load_dataset("cnn_dailymail", "3.0.0"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="article"),
        },
        "creative_writing": {
            "name": "pg19",
            "load_dataset_fn": lambda: load_dataset(
                "pg19"
            ),  # Dataset from this paper: https://arxiv.org/pdf/1911.05507.pdf
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(
                dataset,
                map_batch_size=1,  # PG19 contains very very long texts, so we need to reduce the batch size
            ),
            "cut_off": 3_000,
            "dataset_map_batch_size": 3,
        },
    },
    "german": {
        "wikipedia": {
            "name": "wikipedia",
            # Use dataset processed by Hugging Face to save time
            "load_dataset_fn": lambda: load_dataset("wikipedia", "20220301.de"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, map_batch_size=500),
            "cut_off": 100_000,
            "dataset_map_batch_size": 300,
        },
        "product_reviews": {
            "name": "amazon_reviews_multi",
            # The Multilingual Amazon Reviews Corpus is no longer available on Hugging Face Datasets, we need to use a downloaded version
            "load_dataset_fn": lambda: load_dataset(
                "/ukp-storage-1/englaender/.cache/huggingface/datasets/amazon_reviews_multi/de/1.0.0/"
            ),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="review_body"),
            "dataset_map_batch_size": 300,
        },
        "news": {
            "name": "mlsum",
            "load_dataset_fn": lambda: load_dataset("mlsum", "de"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset),
            "cut_off": 100_000,
        },
        "creative_writing": {
            "name": "opus & corpus_of_german_fiction_txt",
            "load_dataset_fn": lambda: load_german_creative_writing(),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, map_batch_size=1),
            "dataset_map_batch_size": 50,
        },
    },
    "chinese": {
        "wikipedia": {
            "name": "wikipedia",
            "load_dataset_fn": lambda: load_dataset("wikimedia/wikipedia", "20231101.zh"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, map_batch_size=100),
            "cut_off": 100_000,
            "dataset_map_batch_size": 500,
        },
        "product_reviews": {
            "name": "amazon_reviews_multi",
            # The Multilingual Amazon Reviews Corpus is no longer available on Hugging Face Datasets, we need to use a downloaded version
            "load_dataset_fn": lambda: load_dataset(
                "/ukp-storage-1/englaender/.cache/huggingface/datasets/amazon_reviews_multi/zh/1.0.0/"
            ),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(
                dataset, text_field="review_body", map_batch_size=100
            ),
            "dataset_map_batch_size": 500,
        },
        "news": {
            "name": "xl-sum",
            "load_dataset_fn": lambda: load_dataset("csebuetnlp/xlsum", "chinese_simplified"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="text"),
        },
        "creative_writing": None,
    },
    "turkish": {
        "wikipedia": {
            "name": "wikipedia",
            "load_dataset_fn": lambda: load_dataset("wikimedia/wikipedia", "20231101.tr"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, map_batch_size=100),
            "cut_off": 100_000,
            "dataset_map_batch_size": 500,
        },
        "product_reviews": {
            "name": "turkish_product_reviews",
            "load_dataset_fn": lambda: load_dataset("turkish_product_reviews"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="sentence"),
        },
        "news": {
            "name": "xl-sum",
            "load_dataset_fn": lambda: load_dataset("csebuetnlp/xlsum", "turkish"),
            "tokenize_fn": lambda dataset: standard_preprocess_dataset(dataset, text_field="text"),
        },
        "creative_writing": None,
    },
}


def load_german_creative_writing():
    # Combine Corpus of German-Language Fiction & Opus Books to get more data
    opus_books = load_opus_books(
        language="de",
        subset_names=[
            "ca-de",
            "de-en",
            "de-eo",
            "de-es",
            "de-fr",
            "de-hu",
            "de-it",
            "de-nl",
            "de-pt",
            "de-ru",
        ],
    )

    corpus_of_german_fiction_txt = load_corpus_of_german_fiction_txt()

    dataset = DatasetDict(
        {
            "train": concatenate_datasets([opus_books, corpus_of_german_fiction_txt]),
        }
    )

    return dataset


def load_corpus_of_german_fiction_txt() -> Dataset:
    # Corpus of German-Language Fiction
    # https://ds.ifi.uni-heidelberg.de/files/Team/jannik/publications/fischer-stroetgen_temporal-expressions-in-literary-corpora_dh2015_final_2015-03-01.pdf
    # https://figshare.com/articles/dataset/Corpus_of_German-Language_Fiction_txt_/4524680

    # 1. Load every file in the corpus folder
    texts = []
    for file in Path("Corpora/corpus-of-german-fiction-txt").glob("*.txt"):
        with open(file, "r") as f:
            # Read the file and remove newlines
            text = f.read().replace("\n", " ")
            texts.append(text)

    # Create Dataset
    return Dataset.from_dict({"text": texts})


def load_opus_books(language, subset_names: list) -> Dataset:
    # This function creates dataset from the opus_books dataset
    # opus_books is a collection of translated sentences that are in correct order. So we can:

    mapped_datasets = []
    for subset_name in subset_names:
        dataset = load_dataset("opus_books", subset_name)
        dataset = dataset.map(
            lambda example: {"text": example["translation"][language]},
            remove_columns=["id", "translation"],
        )

        # The different translation pairs are in correct order, so we can concatenate them to get the full text
        dataset = " ".join(dataset["train"]["text"])  # Concatenate all texts

        mapped_datasets.append(Dataset.from_dict({"text": [dataset]}))

    return concatenate_datasets([dataset for dataset in mapped_datasets])


def standard_preprocess_dataset(dataset: DatasetDict, text_field="text", map_batch_size=1000):
    """
    This function tokenizes the dataset and returns it.
    """
    return dataset.map(
        lambda examples: tokenizer([" ".join(x) for x in examples[text_field]], return_special_tokens_mask=True),
        batched=True,
        batch_size=map_batch_size,
        num_proc=1,  # We get errors on slurm if we use more than 1 worker
        remove_columns=dataset["train"].column_names,
    )


# This function is from Hugging Face, didn't change anything
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


def load_and_preprocess_dataset(domain, language):
    """
    This function preprocesses the dataset and returns it.
    If the dataset was already preprocessed in a previous run, it will be loaded from disk (this is much faster)
    """
    dataset_information = DATASETS[language][domain]
    dataset_save_path = PREPROCESSED_DATASET_PATH / language / domain

    # Check if dataset was already preprocessed in a previous run
    if (dataset_save_path).exists():
        print(f"Loading preprocessed dataset {dataset_information['name']}")
        dataset = load_from_disk(dataset_save_path)

    else:
        print(f"Loading and preprocessing dataset {dataset_information['name']}")
        dataset: Dataset = dataset_information["load_dataset_fn"]()
        dataset = dataset.shuffle(DATASET_SHUFFLE_SEED)

        # Some datasets are very large, so we only use a subset of them (after shuffling)
        if "cut_off" in dataset_information:
            dataset["train"] = dataset["train"].select(range(dataset_information["cut_off"]))

        # Tokenize + group texts into chunks of 512 tokens
        print(f"Running tokenizer on every text in dataset {dataset_information['name']}:")
        dataset = dataset_information["tokenize_fn"](dataset)
        print("Finished tokenizing. Grouping texts into chunks of 512 tokens:")
        dataset = dataset.map(
            group_texts,
            batched=True,
            batch_size=dataset_information.get("dataset_map_batch_size", 1000),
            num_proc=1,  # We get errors on slurm if we use more than 1 worker
        )

        # Shuffle again + save
        dataset["train"] = dataset["train"].shuffle(DATASET_SHUFFLE_SEED)
        dataset.save_to_disk(dataset_save_path)

    total_steps_per_dataset = STEPS_PER_LANGUAGE_DOMAIN[language][domain] * BATCH_SIZE

    # Throw away examples we will not use during training
    if dataset["train"].num_rows > total_steps_per_dataset:
        dataset["train"] = dataset["train"].select(range(total_steps_per_dataset))

    # This probably makes the training slightly faster (?)
    dataset["train"] = dataset["train"].flatten_indices()

    print(f"For {domain}-{language} we have {dataset['train'].num_rows} rows and train for {total_steps_per_dataset} steps => Epochs trained on dataset: {total_steps_per_dataset / dataset['train'].num_rows}\n")  # fmt: skip

    return dataset


def load_datasets():
    train_datasets = []
    for language in DATASETS:
        if language not in LANGUAGES:
            continue
        for domain in DATASETS[language]:
            if domain not in DOMAINS:
                continue
            if DATASETS[language][domain] is None:
                continue

            print(f"Loading dataset for {domain}-{language}")

            dataset = load_and_preprocess_dataset(language=language, domain=domain)
            train_datasets.append(
                {
                    "domain": domain,
                    "language": language,
                    "dataset": dataset["train"],
                }
            )

    return train_datasets
