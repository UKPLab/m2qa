import pathlib

MODEL_NAME = "xlm-roberta-base"
CHUNK_SIZE = 512  # is xlm-r tokenizer model_max_length

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1  # values > 1 are not yet supported in DomainLanguageTrainer
NUM_WORKERS = 8

SEED = 123
DATASET_SHUFFLE_SEED = 42

# Paths
SAVE_ADAPTER_PATH = pathlib.Path(
    "/storage/ukp/work/englaender/M2QA/m2qa/Experiments/storage/Trained_adapters/pure_domain_language/"
)
OUTPUT_DIR = pathlib.Path("/storage/ukp/work/englaender/M2QA/m2qa/Experiments/storage/pure_domain_language/")
PREPROCESSED_DATASET_PATH = pathlib.Path(
    "/storage/ukp/work/englaender/M2QA/m2qa/Experiments/storage/Preprocessed_datasets"
)

# Domains and languages we want to train adapters for
LANGUAGES = ["english", "german", "chinese", "turkish"]
DOMAINS = ["wikipedia", "product_reviews", "news", "creative_writing"]

# Steps
STEPS_PER_LANGUAGE_DOMAIN = {
    # Training every adapter for a total of 62500 update steps => total of 250000 steps
    "english": {
        "wikipedia": 10417,
        "product_reviews": 10416,
        "news": 10417,
        "creative_writing": 31250,
    },
    "german": {
        "wikipedia": 10417,
        "product_reviews": 10416,
        "news": 10417,
        "creative_writing": 31250,
    },
    "chinese": {
        "wikipedia": 20833,
        "product_reviews": 20833,
        "news": 20833,
        # doesn't have creative writing
    },
    "turkish": {
        "wikipedia": 20833,
        "product_reviews": 20833,
        "news": 20833,
        # doesn't have creative writing
    },
}
