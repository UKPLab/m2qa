import argparse
import json
import pathlib
from collections import defaultdict
from datasets import load_dataset
import random
import sys

sys.path.append("../")
from M2QA_Metric.m2qa_metric import M2QAMetric
import pprint

m2qa_metric = M2QAMetric()

LANGUAGES = ["german", "chinese", "turkish"]
DOMAINS = ["news", "creative_writing", "product_reviews"]


def load_m2qa_dataset():
    m2qa_dataset = {}
    for language in ["german", "chinese", "turkish"]:
        domains = ["news", "creative_writing", "product_reviews"]
        m2qa_dataset[language] = load_dataset(
            "json",
            data_files={domain: f"../../m2qa_dataset/{language}/{domain}.json" for domain in domains},
        )

    return m2qa_dataset


def answerable_unanswerable(
    predictions: dict[str, str], references: dict[str, str], keep_only_answerable: bool
) -> tuple[dict[str, str], dict[str, str]]:
    filtered_predictions = []
    filtered_references = []
    for index, value in enumerate(references):
        if keep_only_answerable and len(value["answers"]["text"]) == 0:
            continue
        if not keep_only_answerable and len(value["answers"]["text"]) != 0:
            continue
        filtered_predictions.append(predictions[index])
        filtered_references.append(value)

    return filtered_predictions, filtered_references


def evaluate_predictions(
    predictions_file_name,
    references_file_name,
    language,
    domain,
    misclassified_examples: dict[str, list[str]] = None,
    low_f1_examples: dict[str, list[str]] = None,
    low_f1_or_missclassified_examples: dict[str, list[str]] = None,
    answer_string_in_all_languages: bool = False,
    m2qa_dataset=None,
):
    with open(predictions_file_name, "r") as predictions_file:
        predictions = json.load(predictions_file)

    # Preprocess predictions
    for value in predictions:
        if answer_string_in_all_languages:
            if value["prediction_text"] in ["unbeantwortbar", "无法回答", "cevaplanamaz"]:
                value["no_answer_probability"] = 1.0
            else:
                value["no_answer_probability"] = 0.0

        else:
            if value["prediction_text"] == "unanswerable":
                value["no_answer_probability"] = 1.0
            else:
                value["no_answer_probability"] = 0.0

    with open(references_file_name, "r") as references_file:
        references = json.load(references_file)

    print(f"{language} - {domain}")

    # create map that maps the example id to the index in the references list
    references_map = {example["id"]: index for index, example in enumerate(references)}
    example_map = {example["id"]: example for example in m2qa_dataset[language][domain]}

    for value in predictions:
        reference = references[references_map[value["id"]]]

        if value["no_answer_probability"] == 1.0 and len(reference["answers"]["text"]) == 0:
            # Correct classification of unanwerable questions
            pass

        elif value["no_answer_probability"] == 0.0 and len(reference["answers"]["text"]) != 0:
            # Correct classification of answerable questions
            # => Compute F1 score
            results = m2qa_metric.compute(
                predictions=[value],
                references=[reference],
                no_answer_threshold=0.95,
                language=language,
            )

            # Check if F1 score is lower than 25
            if results["f1"] < 25:
                example_id = value["id"]
                example = example_map[example_id]

                answer = {
                    "model": str(predictions_file_name),
                    "f1": results["f1"],
                    "prediction": value["prediction_text"],
                    "expected": reference["answers"]["text"],
                    "question": example["question"],
                    "context": example["context"],
                }

                low_f1_examples[value["id"]].append(answer)
                low_f1_or_missclassified_examples[value["id"]].append(answer)

        else:
            # Misclassified example
            example_id = value["id"]
            example = example_map[example_id]
            answer = {
                "model": str(predictions_file_name),
                "prediction": value["prediction_text"],
                "expected": reference["answers"]["text"],
                "question": example["question"],
                "context": example["context"],
            }
            misclassified_examples[value["id"]].append(answer)
            low_f1_or_missclassified_examples[value["id"]].append(answer)


def main(args: argparse.Namespace):
    misclassified_examples = defaultdict(list)
    low_f1_examples = defaultdict(list)
    low_f1_or_missclassified_examples = defaultdict(list)
    m2qa_dataset = load_m2qa_dataset()

    for input_dir in args.input_dirs:
        if args.dont_skip_squad_xquad:
            # SQuAD
            predictions_file_name = pathlib.Path(input_dir) / "squad_v2" / "predictions.json"
            references_file_name = pathlib.Path(input_dir) / "squad_v2" / "references.json"
            if predictions_file_name.exists():
                evaluate_predictions(
                    predictions_file_name=predictions_file_name,
                    references_file_name=references_file_name,
                    language="english",
                    domain="squad_v2",
                )
            else:
                print(f"Skipping SQuAD because the file {predictions_file_name} does not exist.")

            # XQuAD
            for language in ["english", "german", "chinese", "turkish"]:
                predictions_file_name = pathlib.Path(input_dir) / "xquad" / language / "predictions.json"
                references_file_name = pathlib.Path(input_dir) / "xquad" / language / "references.json"
                if predictions_file_name.exists():
                    evaluate_predictions(
                        predictions_file_name=predictions_file_name,
                        references_file_name=references_file_name,
                        language=language,
                        domain="xquad",
                    )
                else:
                    print(f"Skipping XQuAD {language} because the file {predictions_file_name} does not exist.")

        # M2QA
        for language in ["german"]:
            print("For this analysis we only analyse german examples.")
            for domain in DOMAINS:
                if not (pathlib.Path(input_dir) / language / domain).exists():
                    print(f"Skipping {language} {domain} because the directory does not exist.")
                else:
                    predictions_file_name = pathlib.Path(input_dir) / language / domain / "predictions.json"
                    references_file_name = pathlib.Path(input_dir) / language / domain / "references.json"
                    evaluate_predictions(
                        predictions_file_name,
                        references_file_name,
                        language,
                        domain,
                        answer_string_in_all_languages=args.answer_string_in_all_languages,
                        misclassified_examples=misclassified_examples,
                        low_f1_examples=low_f1_examples,
                        low_f1_or_missclassified_examples=low_f1_or_missclassified_examples,
                        m2qa_dataset=m2qa_dataset,
                    )

    misclassified_examples_ids = [
        example_id for example_id, models in misclassified_examples.items() if len(models) == len(args.input_dirs)
    ]
    low_f1_examples_ids = [
        example_id for example_id, models in low_f1_examples.items() if len(models) == len(args.input_dirs)
    ]
    low_f1_or_missclassified_examples_ids = [
        example_id
        for example_id, models in low_f1_or_missclassified_examples.items()
        if len(models) == len(args.input_dirs)
    ]

    print(f"Number of isolated misclassified examples: {len(misclassified_examples_ids)}")
    print(f"Number of isolated low F1 examples: {len(low_f1_examples_ids)}\n")
    print(f"Raw Number of misclassified examples: {len(misclassified_examples)}")
    print(f"Raw Number of low F1 examples: {len(low_f1_examples)}\n")

    print(f"Number of low F1 or misclassified examples: {len(low_f1_or_missclassified_examples_ids)}\n")
    print(f"Raw Number of low F1 or misclassified examples: {len(low_f1_or_missclassified_examples)}\n")

    # Save the selected examples to a file
    with open("Results/Analysis/misclassified_examples.json", "w") as file:
        random_examples = random.sample(misclassified_examples_ids, k=50)
        selected_examples = {example_id: misclassified_examples[example_id] for example_id in random_examples}

        # For every example: Pull context, question and expected answer only from the first entry, then add for each model the prediction, f1 and name
        for example_id, example in selected_examples.items():
            selected_examples[example_id] = {
                "context": example[0]["context"],
                "question": example[0]["question"],
                "expected answer": example[0]["expected"],
                "models": [
                    {
                        "model": model["model"],
                        "prediction": model["prediction"],
                    }
                    for model in example
                ],
            }

        pprint.pprint(selected_examples, file, width=200)

    with open("Results/Analysis/low_f1_examples.json", "w") as file:
        random_examples = random.sample(low_f1_examples_ids, k=50)
        selected_examples = {example_id: low_f1_examples[example_id] for example_id in random_examples}
        for example_id, example in selected_examples.items():
            selected_examples[example_id] = {
                "context": example[0]["context"],
                "question": example[0]["question"],
                "expected answer": example[0]["expected"],
                "models": [
                    {
                        "model": model["model"],
                        "prediction": model["prediction"],
                        "f1": model["f1"],
                    }
                    for model in example
                ],
            }
        pprint.pprint(selected_examples, file, width=200)

    with open("Results/Analysis/low_f1_OR_missclassified_examples.json", "w") as file:
        selected_examples = {
            example_id: low_f1_or_missclassified_examples[example_id]
            for example_id in low_f1_or_missclassified_examples_ids
        }
        for example_id, example in selected_examples.items():
            selected_examples[example_id] = {
                "context": example[0]["context"],
                "question": example[0]["question"],
                "expected answer": example[0]["expected"],
                "models": [
                    {
                        "model": model["model"],
                        "prediction": model["prediction"],
                        "f1": model["f1"] if "f1" in model else None,
                    }
                    for model in example
                ],
            }
        pprint.pprint(selected_examples, file, width=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dont_skip_squad_xquad", type=bool, default=True, const=False, nargs="?")
    parser.add_argument(
        "--input_dirs",
        required=True,
        type=str,
        nargs="+",
        help="List of directories containing the predictions. You need to give 4 direcotries (2 models x 2 promps [zero-shot, five-shot])",
    )
    parser.add_argument("--answer_string_in_all_languages", type=bool, default=False, const=True, nargs="?")

    main(parser.parse_args())
