import os

from trainer import DomainLanguageTrainer
from hyper_parameter import (
    BATCH_SIZE,
    DOMAINS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGES,
    MODEL_NAME,
    NUM_WORKERS,
    OUTPUT_DIR,
    PREPROCESSED_DATASET_PATH,
    SAVE_ADAPTER_PATH,
    SEED,
    STEPS_PER_LANGUAGE_DOMAIN,
)
from load_datasets import load_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PfeifferConfig,
    TrainingArguments,
    XLMRobertaAdapterModel,
    set_seed,
)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def save_adapters(model: XLMRobertaAdapterModel):
    path_save_head = SAVE_ADAPTER_PATH / "head"
    path_save_head.mkdir(parents=True, exist_ok=True)
    path_save_domain = SAVE_ADAPTER_PATH / "domain"
    path_save_domain.mkdir(parents=True, exist_ok=True)
    path_save_language = SAVE_ADAPTER_PATH / "language"
    path_save_language.mkdir(parents=True, exist_ok=True)

    # 4.1 Save domain adapters
    for domain in DOMAINS:
        model.save_adapter(path_save_domain / domain, domain, with_head=False)

    # 4.2 Save language adapters
    for language in LANGUAGES:
        model.save_adapter(path_save_language / language, language, with_head=False)


def main():
    # Set seed before initializing model.
    set_seed(SEED)

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and data_collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    lm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.4,  # XLM-R has a comparable size to bert-large and thus 0.4 is probably the best mlm_probability: https://arxiv.org/pdf/2202.08005.pdf
    )

    # 1. Load model
    print("Loading model...")
    model = XLMRobertaAdapterModel.from_pretrained(MODEL_NAME)

    # 2. Add domain & language adapters to model
    print("Adding domain & language adapters to model...")
    for language in LANGUAGES:
        config = PfeifferConfig(reduction_factor=2)
        model.add_adapter(language, config=config)

    for domain in DOMAINS:
        config = PfeifferConfig(reduction_factor=2)
        model.add_adapter(domain, config=config)

    # 3. Add MLM head
    print("Adding MLM head to model...")
    model.add_masked_lm_head("mlm_head")
    config = PfeifferConfig(reduction_factor=2)
    # model.add_adapter("mlm_head", config=config)

    print(f"model after adding adapters:\n{model.adapter_summary()}")

    # 4. Load datasets
    print("Loading datasets...")
    train_datasets = load_datasets()

    # 5. Train model
    training_args = TrainingArguments(
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=1e-4,
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        report_to="none",
        dataloader_num_workers=NUM_WORKERS,
        save_steps=5000,
        logging_steps=1000,
        warmup_steps=1000,
        lr_scheduler_type="linear",  # cosine
    )

    # need to set head active before calling AdapterTrainer.__init__ in DomainLanguageTrainer
    model.set_active_adapters(LANGUAGES[0])

    trainer = DomainLanguageTrainer(
        model=model,
        args=training_args,
        train_datasets=train_datasets,
        eval_datasets=None,
        steps_per_dataset_dict=STEPS_PER_LANGUAGE_DOMAIN,
        mlm_head_name="mlm_head",
        tokenizer=tokenizer,
        data_collator=lm_data_collator,
    )

    trainer.train()

    # 4. Save the adapters
    save_adapters(model)


if __name__ == "__main__":
    main()
