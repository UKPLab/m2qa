import argparse
import json
import time
from pathlib import Path

from OpenAI.api_request_parallel_processor import run_openai_requests
from OpenAI.prepare_files_for_generation import prepare_files_for_generation_openai
from prompts import PROMPTS

MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE = {
    "german": [
        "news",
        "creative_writing",
        "product_reviews",
    ],
    "chinese": [
        "news",
        "creative_writing",
        "product_reviews",
    ],
    "turkish": [
        "news",
        "creative_writing",
        "product_reviews",
    ],
}


def predictions_raw_to_json(
    predictions_filepath_raw: Path,
    predictions_filepath: Path,
    references_filepath: Path,
):
    # Predictions_filepath_raw contains a jsonl file with the predictions
    # These are in a different order than the references (because of the parallel processing)
    # Transform the raw results into a json file by reading each line and appending it to a list
    # The list is then sorted by the id in the same order as the references id list
    predictions = {}
    with open(predictions_filepath_raw, "r") as f:
        for line in f:
            prediction = json.loads(line)
            id = prediction["id"]
            predictions[id] = prediction

    with open(references_filepath, "r") as references_file:
        references = json.load(references_file)

    references_example_ids = [reference["id"] for reference in references]

    # Sort the predictions by the id in the same order as the references id list
    predictions_sorted = []
    for id in references_example_ids:
        predictions_sorted.append(predictions[id])

    # Check if there are any predictions missing
    if len(predictions_sorted) != len(references):
        raise Exception("Not all predictions were found.")

    # Save the predictions as a json file
    predictions_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_filepath, "w") as f:
        json.dump(predictions_sorted, f, ensure_ascii=False)


def main(prompt_name, model_name, output_dir, evaluate_squad_and_xquad=False):
    prompt_fn = PROMPTS[prompt_name]
    directory = Path(output_dir) / model_name / prompt_name

    # Check if the directory exists
    if directory.exists():
        raise Exception("The directory already exists. Please provide a new one, so that no data is overwritten.")

    # Read values for the API from api.json
    with open("OpenAI/api.json") as f:
        api_data = json.load(f)

    # 1. call "prepare_files_for_generation.py"
    print("OPENAI prepare files for generation")
    prepare_files_for_generation_openai(
        model=model_name,
        output_dir=directory,
        languages_and_domains_to_evaluate=MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE,
        prompt_fn=prompt_fn,
        prompt_name=prompt_name,
        evaluate_squad_and_xquad=evaluate_squad_and_xquad,
    )

    api_base = api_data["api_base"]
    api_version = api_data["api-version"]
    request_url = f"{api_base}openai/deployments/{model_name}/chat/completions?api-version={api_version}"

    if evaluate_squad_and_xquad:
        # SQuAD
        predictions_filepath_raw = directory / "squad_v2" / "predictions_raw.jsonl"
        print(f"OPENAI generate answers for: SQuADv2, request_url: {request_url}")

        run_openai_requests(
            requests_filepath=directory / "squad_v2" / "messages.jsonl",
            save_filepath=predictions_filepath_raw,
            request_url=request_url,
            api_key=api_data["api_key"],
            max_requests_per_minute=api_data["max_requests_per_minute"],
            max_tokens_per_minute=api_data["max_tokens_per_minute"],
        )

        predictions_raw_to_json(
            predictions_filepath_raw=predictions_filepath_raw,
            predictions_filepath=directory / "squad_v2" / "predictions.json",
            references_filepath=directory / "squad_v2" / "references.json",
        )

        # XQuAD
        for language in ["english", "german", "chinese", "turkish"]:
            print("Sleeping for 60 seconds to avoid going over the rate limit.")
            time.sleep(60)
            predictions_filepath_raw = directory / "xquad" / language / "predictions_raw.jsonl"
            print(f"OPENAI generate answers for XQuAD: language: {language}, request_url: {request_url}")

            run_openai_requests(
                requests_filepath=directory / "xquad" / language / "messages.jsonl",
                save_filepath=predictions_filepath_raw,
                request_url=request_url,
                api_key=api_data["api_key"],
                max_requests_per_minute=api_data["max_requests_per_minute"],
                max_tokens_per_minute=api_data["max_tokens_per_minute"],
            )

            predictions_raw_to_json(
                predictions_filepath_raw=predictions_filepath_raw,
                predictions_filepath=directory / "xquad" / language / "predictions.json",
                references_filepath=directory / "xquad" / language / "references.json",
            )

    # M2QA
    for language in MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE:
        domains = MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE[language]
        for domain in domains:
            print("Sleeping for 60 seconds to avoid going over the rate limit.")
            time.sleep(60)
            predictions_filepath_raw = directory / language / domain / "predictions_raw.jsonl"
            print(f"OPENAI generate answers for: language: {language}, domain: {domain}, request_url: {request_url}")

            run_openai_requests(
                requests_filepath=directory / language / domain / "messages.jsonl",
                save_filepath=predictions_filepath_raw,
                request_url=request_url,
                api_key=api_data["api_key"],
                max_requests_per_minute=api_data["max_requests_per_minute"],
                max_tokens_per_minute=api_data["max_tokens_per_minute"],
            )

            predictions_raw_to_json(
                predictions_filepath_raw=predictions_filepath_raw,
                predictions_filepath=directory / language / domain / "predictions.json",
                references_filepath=directory / language / domain / "references.json",
            )

    # Evaluate the results seperatly
    print("\n\n\nFinished generating the results.")
    print("Results are in the directory: " + str(directory))
    print(f'Execute this command to evaluate: python3 evaluate_llm_answers.py --input_dir="{directory}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompt_name", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--evaluate_squad_and_xquad", type=bool, default=False, const=True, nargs="?")  # fmt: skip
    args = parser.parse_args()

    # Check if the model is valid
    if args.model not in ["gpt-35-turbo-0301", "gpt-35-turbo-0613"]:
        raise Exception(
            "The model is not valid. Please provide a valid model (gpt-35-turbo-0301 or gpt-35-turbo-0613)"
        )

    # Check if the prompt is valid
    if args.prompt_name not in PROMPTS:
        raise Exception("The prompt is not valid. Please provide a valid prompt: " + str(PROMPTS.keys()))

    main(args.prompt_name, args.model, args.output_dir, args.evaluate_squad_and_xquad)
