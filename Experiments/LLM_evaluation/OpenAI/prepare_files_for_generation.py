import json
import pathlib
from typing import List

from datasets import DatasetDict, load_dataset

XQUAD_LANGUAGE_MAPPING = {
    "german": "de",
    "english": "en",
    "chinese": "zh",
    "turkish": "tr",
}


def load_mmqa_dataset(languages_and_domains_to_evaluate: dict[str, list[str]], limit: int) -> dict[str, DatasetDict]:
    mmqa_dataset = {}
    for language in languages_and_domains_to_evaluate:
        domains = languages_and_domains_to_evaluate[language]
        mmqa_dataset[language] = load_dataset(
            "json",
            data_files={domain: f"../../m2qa_dataset/{language}/{domain}.json" for domain in domains},
        )

        if limit is not None:
            for domain in mmqa_dataset[language]:
                mmqa_dataset[language][domain] = mmqa_dataset[language][domain].shuffle(seed=42).select(range(limit))

    return mmqa_dataset


def generate(
    model: str,
    output_dir: pathlib.Path,
    prompt_fn,
    prompt_name: str,
    languages_and_domains_to_evaluate,
    limit,
    evaluate_squad_and_xquad,
):
    if evaluate_squad_and_xquad:
        # SQuAD requests
        squad_v2_dataset = load_dataset("squad_v2", split="validation")

        prepare_file_for_dataset(
            model,
            squad_v2_dataset,
            prompt_fn,
            pathlib.Path(output_dir) / "squad_v2",
            prompt_name,
            limit=limit,
        )

        # XQuAD requests
        for language in ["english", "german", "chinese", "turkish"]:
            xquad_dataset = load_dataset(
                "xquad",
                f"xquad.{XQUAD_LANGUAGE_MAPPING[language]}",
                revision="8c2924a720ea543c2b6346284e21d3b85b1c2996",
            )

            prepare_file_for_dataset(
                model,
                xquad_dataset["validation"],
                prompt_fn,
                pathlib.Path(output_dir) / "xquad" / language,
                prompt_name,
                limit=limit,
            )

    # M2QA requests
    mmqa_dataset: dict[str, DatasetDict] = load_mmqa_dataset(languages_and_domains_to_evaluate, limit)
    for language, datasetdict in mmqa_dataset.items():
        for domain, dataset in datasetdict.items():
            prepare_file_for_dataset(
                model,
                dataset,
                prompt_fn,
                pathlib.Path(output_dir) / language / domain,
                prompt_name,
                domain=domain,
                language=language,
                limit=limit,
            )


def prepare_file_for_dataset(
    model,
    dataset,
    prompt_fn,
    output_dir: pathlib.Path,
    prompt_name: str,
    domain: str = None,
    language: str = None,
    limit=None,
):
    requests_list = []

    for example in dataset:
        context = example["context"]
        question = example["question"]

        if prompt_name == "five_shot_cross_lingual" or prompt_name == "five_shot_cross_domain":
            messages = prompt_fn[language][domain](context, question)
        elif prompt_name == "zero_shot_same_language":
            messages = prompt_fn[language](context, question)
        else:
            messages = prompt_fn(context, question)

        requests_list.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": 50,
                "temperature": 0.0,
                "prompt_used": prompt_name,
                "example_id": example["id"],
            }
        )

    # Save the messages that have to be sent to the chatbot
    messages_file_name = output_dir / "messages.jsonl"
    messages_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(messages_file_name, "w") as output_file:
        json.dump(requests_list, output_file, ensure_ascii=False)

    # Save the dataset, though not specifically necessary it makes the evaluation script easier to write
    references_file_name = output_dir / "references.json"
    references_file_name.parent.mkdir(parents=True, exist_ok=True)

    references = [{"id": example["id"], "answers": example["answers"]} for example in dataset]

    with open(references_file_name, "w") as output_file:
        json.dump(references, output_file, ensure_ascii=False)


def prepare_files_for_generation_openai(
    model,
    output_dir,
    languages_and_domains_to_evaluate: dict[str, List[str]],
    prompt_fn,
    prompt_name: str,
    limit=None,
    evaluate_squad_and_xquad=False,
):
    generate(
        model, output_dir, prompt_fn, prompt_name, languages_and_domains_to_evaluate, limit, evaluate_squad_and_xquad
    )
