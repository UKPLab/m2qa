import argparse
import json
import pathlib
import evaluate

import sys

sys.path.append("../")
from M2QA_Metric.m2qa_metric import M2QAMetric

USE_IMPROVED_METRIC = False

if USE_IMPROVED_METRIC:
    m2qa_metric = M2QAMetric()
else:
    m2qa_metric = evaluate.load("squad_v2")

LANGUAGES = ["german", "chinese", "turkish"]
DOMAINS = ["news", "creative_writing", "product_reviews"]


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
    anaylse_answerable_vs_unanswerable: bool = True,
    answer_string_in_all_languages: bool = False,
):
    with open(predictions_file_name, "r") as predictions_file:
        predictions = json.load(predictions_file)

    # Preprocess predictions
    # set no_answer_probability to 1.0 if the answer "unbeantwortbar", "无法回答" or "cevaplanamaz"
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

    print(f"========================== {language} -" f" {domain} ==========================")
    if USE_IMPROVED_METRIC:
        full_results = m2qa_metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=0.95,
            language=language,
        )
    else:
        full_results = m2qa_metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=0.95,
        )
    # Evaluate accuracy of unanswerable classification
    unanswerable_classification_correct = 0
    total_examples = 0

    # create map that maps the example id to the index in the references list
    references_map = {example["id"]: index for index, example in enumerate(references)}

    for value in predictions:
        reference = references[references_map[value["id"]]]

        # Correct classification of unanwerable questions
        if value["no_answer_probability"] == 1.0 and len(reference["answers"]["text"]) == 0:
            unanswerable_classification_correct += 1

        # Correct classification of answerable questions
        if value["no_answer_probability"] == 0.0 and len(reference["answers"]["text"]) != 0:
            unanswerable_classification_correct += 1

        total_examples += 1

    classification_accuracy = unanswerable_classification_correct / total_examples * 100

    # All results
    print(f"Full results for {language} {domain}:")
    print(f"Classification accuracy: {classification_accuracy:.2f}\\%")
    print(f"F1 / EM: {full_results['f1']:.2f} / {full_results['exact']:.2f}")
    print(full_results)

    if anaylse_answerable_vs_unanswerable:
        # Answerable results
        answerable_predictions, answerable_references = answerable_unanswerable(predictions, references, True)
        if USE_IMPROVED_METRIC:
            answerable_results = m2qa_metric.compute(
                predictions=answerable_predictions,
                references=answerable_references,
                no_answer_threshold=0.95,
                language=language,
            )
        else:
            answerable_results = m2qa_metric.compute(
                predictions=answerable_predictions,
                references=answerable_references,
                no_answer_threshold=0.95,
            )
        print(f"\nAnswerable results for {language} - {domain}")
        print(f"F1 / EM: {answerable_results['f1']:.2f} / {answerable_results['exact']:.2f}")
        print(answerable_results)

        # Unanswerable results
        unanswerable_predictions, unanswerable_references = answerable_unanswerable(predictions, references, False)
        if USE_IMPROVED_METRIC:
            unanswerable_results = m2qa_metric.compute(
                predictions=unanswerable_predictions,
                references=unanswerable_references,
                no_answer_threshold=0.95,
                language=language,
            )
        else:
            unanswerable_results = m2qa_metric.compute(
                predictions=unanswerable_predictions,
                references=unanswerable_references,
                no_answer_threshold=0.95,
            )
        print(f"\nUnanswerable results for {language} - {domain}")
        print(f"F1 / EM: {unanswerable_results['f1']:.2f} / {unanswerable_results['exact']:.2f}")
        print(unanswerable_results)


def main(args: argparse.Namespace):
    # SQuAD
    predictions_file_name = pathlib.Path(args.input_dir) / "squad_v2" / "predictions.json"
    references_file_name = pathlib.Path(args.input_dir) / "squad_v2" / "references.json"
    if predictions_file_name.exists():
        evaluate_predictions(predictions_file_name, references_file_name, "english", "squad_v2")
    else:
        print(f"Skipping SQuAD because the file {predictions_file_name} does not exist.")

    # XQuAD
    for language in ["english", "german", "chinese", "turkish"]:
        predictions_file_name = pathlib.Path(args.input_dir) / "xquad" / language / "predictions.json"
        references_file_name = pathlib.Path(args.input_dir) / "xquad" / language / "references.json"
        if predictions_file_name.exists():
            evaluate_predictions(predictions_file_name, references_file_name, language, "xquad", False)
        else:
            print(f"Skipping XQuAD {language} because the file {predictions_file_name} does not exist.")

    # M2QA
    for language in LANGUAGES:
        for domain in DOMAINS:
            if not (pathlib.Path(args.input_dir) / language / domain).exists():
                print(f"Skipping {language} {domain} because the directory does not exist.")
            else:
                predictions_file_name = pathlib.Path(args.input_dir) / language / domain / "predictions.json"
                references_file_name = pathlib.Path(args.input_dir) / language / domain / "references.json"
                evaluate_predictions(
                    predictions_file_name,
                    references_file_name,
                    language,
                    domain,
                    answer_string_in_all_languages=args.answer_string_in_all_languages,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    # If the args.answer_string_in_all_languages is set to true, the evaluation will treat the answer strings "unbeantwortbar", "无法回答" and "cevaplanamaz" as unanswerable answers.
    # Should be used for the isolated cross lingual and isolated cross domain evaluation. (Figures in appendix)
    parser.add_argument("--answer_string_in_all_languages", type=bool, default=False, const=True, nargs="?")  # fmt: skip

    print("=============================================")
    print(
        "IF YOU WANT TO EVALUATE WITH THE IMPROVED METRIC, PLEASE SET THE USE_IMPROVED_METRIC VARIABLE IN LINE 11 TO TRUE"
    )
    print("=============================================")

    main(parser.parse_args())
