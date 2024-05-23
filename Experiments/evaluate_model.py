import argparse
import pprint
from enum import Enum
import os

import evaluate
import torch
from datasets import load_dataset
from evaluate import QuestionAnsweringEvaluator
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    QuestionAnsweringPipeline,
    XLMRobertaAdapterModel,
)
from datasets import Dataset
from transformers.adapters.composition import Stack
import jieba
from M2QA_Metric.m2qa_metric import M2QAMetric

MODEL_NAME = "xlm-roberta-base"
PATHS = {
    # Baseline XLM-R models
    "path_fully_fine_tuned_model": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_model/Thesis_used/xlm_r_fully_finetuned/",
    "paths_xlm_r_domain_adapted": {
        "wiki": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_model/Thesis_used/domain_models/xlm-r-wiki-512-squad",
        "creative_writing": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_model/Thesis_used/domain_models/xlm-r-books-512-squad",
        "news": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_model/Thesis_used/domain_models/xlm-r-news-squad-512-64",
        "product_reviews": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_model/Thesis_used/domain_models/xlm-r-reviews-yelp-squad-512-64",
    },
    # MAD-X+Domain
    "mad-x-domain": {
        "domains": {
            "wiki": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-domain/wiki/",
            "creative_writing": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-domain/books/",
            "news": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-domain/news/",
            "product_reviews": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-domain/reviews/",
        },
        "languages": {
            # The original MAD-X language adapters
            "english": "en/wiki@ukp",
            "german": "de/wiki@ukp",
            "chinese": "zh/wiki@ukp",
            "turkish": "tr/wiki@ukp",
        },
        "qa_head": {
            "name": "mad-x+domain qa_head",
            "path": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-domain/squad2_qa_head",
        },
    },
    # MAD-X²
    "mad-x-2": {
        "domains": {
            "wiki": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/domain/wikipedia/",
            "creative_writing": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/domain/creative_writing/",
            "news": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/domain/news/",
            "product_reviews": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/domain/product_reviews/",
        },
        "languages": {
            "english": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/language/english",
            "german": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/language/german",
            "chinese": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/language/chinese",
            "turkish": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/language/turkish",
        },
        "qa_head": {
            "name": "mad-x-2-qa_adapter",
            "path": "/home/leon/UKP/M2QA/m2qa/Experiments/Trained_adapters/mad-x-2/squad2_qa_head",
        },
    },
}

##########################################################################################
# Other
M2QA_LANGUAGES_AND_DOMAINS_TO_EVALUATE = {
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


XQUAD_LANGUAGE_MAPPING = {
    "german": "de",
    "english": "en",
    "chinese": "zh",
    "turkish": "tr",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_evaluator: QuestionAnsweringEvaluator = evaluate.evaluator("question-answering")  # type: ignore

##########################################################################################
# Helper functions


class ExperimentSetup(str, Enum):
    XLM_R = "xlm-r fine-tuned on SQuADv2"
    XLM_R_DOMAIN_ADAPTED = "domain adapted xlm-r fine-tuned on SQuADv2"
    # MAD-X+Domain
    MAD_X_DOMAIN = "MAD-X+Domain: language adapter + domain adapter + squad head"
    # MAD-X²
    MAD_X_2 = "MAD-X²: language adapter + domain adapter + squad head"

    # MAD-X+Domain intermediate combinations to see effect of language and domain adapters
    MAD_X_DOMAIN_HEAD = "xlm-r + squad head + en adapter + wiki adapter"
    MAD_X_DOMAIN_ONLY_LANGUAGE = "xlm-r + squad head + language adapter + wiki adapter"
    MAD_X_DOMAIN_ONLY_DOMAIN = "xlm-r + squad head + en adapter + domain adapter"


def set_active_adapter_for_setup(model, experiment_setup: ExperimentSetup, language: str = None, domain: str = None):
    match experiment_setup:
        # XLM-R Baselines don't have adapters
        case ExperimentSetup.XLM_R:
            pass
        case ExperimentSetup.XLM_R_DOMAIN_ADAPTED:
            pass

        # MAD-X+Domain
        case ExperimentSetup.MAD_X_DOMAIN:
            model.active_adapters = Stack(
                f"mad-x+domain {language}", f"mad-x+domain {domain}", PATHS["mad-x-domain"]["qa_head"]["name"]
            )
        case ExperimentSetup.MAD_X_DOMAIN_HEAD:
            model.active_adapters = Stack(
                "mad-x+domain english", "mad-x+domain wiki", PATHS["mad-x-domain"]["qa_head"]["name"]
            )
        case ExperimentSetup.MAD_X_DOMAIN_ONLY_LANGUAGE:
            model.active_adapters = Stack(
                f"mad-x+domain {language}", "mad-x+domain wiki", PATHS["mad-x-domain"]["qa_head"]["name"]
            )
        case ExperimentSetup.MAD_X_DOMAIN_ONLY_DOMAIN:
            model.active_adapters = Stack(
                "mad-x+domain english", f"mad-x+domain {domain}", PATHS["mad-x-domain"]["qa_head"]["name"]
            )

        # MAD-X²
        case ExperimentSetup.MAD_X_2:
            model.active_adapters = Stack(
                f"mad-x-2 {language}", f"mad-x-2 {domain}", PATHS["mad-x-2"]["qa_head"]["name"]
            )

        case _:
            raise ValueError(f"Unknown experiment setup: {experiment_setup}")

    if experiment_setup == ExperimentSetup.XLM_R or experiment_setup == ExperimentSetup.XLM_R_DOMAIN_ADAPTED:
        print("Using fully finetuned model")
    else:
        print(f"Using model:\n{model.adapter_summary()}")


def load_adapters(model: XLMRobertaAdapterModel):
    # 1. Load MAD-X+Domain adapters
    model.load_adapter(PATHS["mad-x-domain"]["qa_head"]["path"], load_as=PATHS["mad-x-domain"]["qa_head"]["name"])
    for language, adapter_path in PATHS["mad-x-domain"]["languages"].items():
        model.load_adapter(adapter_path, load_as=f"mad-x+domain {language}")
    for domain, adapter_path in PATHS["mad-x-domain"]["domains"].items():
        model.load_adapter(adapter_path, load_as=f"mad-x+domain {domain}")

    # 2. Load MAD-X² adapters
    model.load_adapter(PATHS["mad-x-2"]["qa_head"]["path"], load_as=PATHS["mad-x-2"]["qa_head"]["name"])
    for language, adapter_path in PATHS["mad-x-2"]["languages"].items():
        model.load_adapter(adapter_path, load_as=f"mad-x-2 {language}")
    for domain, adapter_path in PATHS["mad-x-2"]["domains"].items():
        model.load_adapter(adapter_path, load_as=f"mad-x-2 {domain}")


def load_m2qa_dataset(args: argparse.Namespace):
    m2qa_dataset = {}
    for language in M2QA_LANGUAGES_AND_DOMAINS_TO_EVALUATE:
        domains = M2QA_LANGUAGES_AND_DOMAINS_TO_EVALUATE[language]
        m2qa_dataset[language] = load_dataset(
            "json",
            data_files={domain: f"../m2qa_dataset/{language}/{domain}.json" for domain in domains},
        )

        if args.only_unanswerable:
            for language in m2qa_dataset:
                for domain in m2qa_dataset[language]:
                    m2qa_dataset[language][domain] = m2qa_dataset[language][domain].filter(
                        lambda example: len(example["answers"]["text"]) == 0
                    )

        if args.only_answerable:
            for language in m2qa_dataset:
                for domain in m2qa_dataset[language]:
                    m2qa_dataset[language][domain] = m2qa_dataset[language][domain].filter(
                        lambda example: len(example["answers"]["text"]) != 0
                    )

    return m2qa_dataset


def print_intermediate_results(dataset: str, model_description: str, results: dict):
    print(f" {dataset}: {model_description} ".center(150, "="))
    print(f"F1 / EM: {results['f1']:.2f} / {results['exact']:.2f}")
    print(f"{results}\n\n")


def load_domain_adapted_model(domain: str):
    if domain not in PATHS["paths_xlm_r_domain_adapted"].keys():
        raise ValueError(f"No domain-adapted model available for domain {domain}")
    return AutoModelForQuestionAnswering.from_pretrained(PATHS["paths_xlm_r_domain_adapted"][domain])


def load_xlm_r_model():
    return AutoModelForQuestionAnswering.from_pretrained(PATHS["path_fully_fine_tuned_model"])


def load_adapter_model():
    model = XLMRobertaAdapterModel.from_pretrained(MODEL_NAME)
    load_adapters(model)
    return model


def add_spaces(text, language):
    if language == "chinese":
        # Add whitespaces between words
        return " ".join(jieba.lcut(text))

    else:
        # Don't add whitespaces for non-Chinese texts
        return text


def _evaluate(model, tokenizer, data, experiment_name: str, experiment_setup: ExperimentSetup, args, language, metric):
    prepared_data = qa_evaluator.load_data(data=data, subset=None, split=None)

    print(f"Prepared data for {language}: {prepared_data}")
    if args.add_white_spaces_to_chinese and language == "chinese":
        print(f"loop")
        prepared_data = prepared_data.map(
            lambda example: {
                "id": example["id"],
                "question": example["question"],
                "context": add_spaces(example["context"], language),
                "answers": example["answers"],
            },
            load_from_cache_file=False,
        )

    pipe = qa_evaluator.prepare_pipeline(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        device=(0 if torch.cuda.is_available() else -1),
    )
    qa_evaluator.PIPELINE_KWARGS["handle_impossible_answer"] = True

    metric_inputs, pipe_inputs = qa_evaluator.prepare_data(
        data=prepared_data,
        question_column="question",
        context_column="context",
        id_column="id",
        label_column="answers",
    )
    predictions, perf_results = qa_evaluator.call_pipeline(pipe, batch_size=16, **pipe_inputs)
    predictions = qa_evaluator.predictions_processor(predictions, squad_v2_format=True, ids=prepared_data["id"])

    # remove the white spaces from the predictions
    if args.add_white_spaces_to_chinese and language == "chinese":
        for prediction in predictions["predictions"]:
            prediction["prediction_text"] = prediction["prediction_text"].replace(" ", "")

    total_number_of_questions = 0
    number_of_correctly_classified_questions = 0
    total_of_questions_classified_as_unanswerable = 0
    example_id_map = {example["id"]: i for i, example in enumerate(data)}

    for prediction in predictions["predictions"]:
        total_number_of_questions += 1
        example = data[example_id_map[prediction["id"]]]
        if len(prediction["prediction_text"]) == 0 and len(example["answers"]["text"]) == 0:
            number_of_correctly_classified_questions += 1
        if len(prediction["prediction_text"]) != 0 and len(example["answers"]["text"]) != 0:
            number_of_correctly_classified_questions += 1
        if len(prediction["prediction_text"]) == 0:
            total_of_questions_classified_as_unanswerable += 1

    metric_inputs.update(predictions)

    # Compute metrics from references and predictions
    # use default HuggingFace values
    qa_evaluator.METRIC_KWARGS["language"] = language
    full_result = qa_evaluator.compute_metric(
        metric=metric,
        metric_inputs=metric_inputs,
        strategy="simple",
        confidence_level="0.95",
        n_resamples=9999,
        random_state=None,
    )

    # print_intermediate_results(f"M2QA - {language} - {domain}: ", experiment_setup.value, full_result)
    print_intermediate_results(experiment_name, experiment_setup.value, full_result)
    print(f"Total number of questions: {total_number_of_questions}")
    print(f"Number of correctly classified questions: {number_of_correctly_classified_questions}")
    print(f"Total of questions classified as unanswerable: {total_of_questions_classified_as_unanswerable}")
    print(
        f"Correct classification accuracy: {number_of_correctly_classified_questions / total_number_of_questions * 100:.2f}%\n\n"
    )

    return full_result


##########################################################################################
# Evaluation functions: SQuADv2, XQuAD, M2QA


def _check_domains_to_evaluate(domains_to_evaluate, domains_list):
    # If domains_to_evaluate is None, return domains_list
    # If domains_to_evaluate is not None, check if all the domains are in domains_list
    # If not, print a warning and return the intersection of domains_to_evaluate and domains_list
    if domains_to_evaluate is None:
        return domains_list
    else:
        domains_to_evaluate = set(domains_to_evaluate)
        domains_list = set(domains_list)

        if not domains_to_evaluate.issubset(domains_list):
            print(
                f"Warning: The domains to evaluate {domains_to_evaluate} are not a subset of the available domains {domains_list}."
            )
            return domains_to_evaluate.intersection(domains_list)
        else:
            return domains_to_evaluate


def evaluate_on_m2qa(
    experiment_setup: ExperimentSetup,
    model,
    dataset,
    all_results_dict: dict,
    args,
    metric,
    domains_to_evaluate=None,  # needed for the domain-adapted XLM-R models
):
    all_results = {}

    for language in M2QA_LANGUAGES_AND_DOMAINS_TO_EVALUATE.keys():
        all_results[language] = {}
        domains_list = M2QA_LANGUAGES_AND_DOMAINS_TO_EVALUATE[language]
        domains = _check_domains_to_evaluate(domains_to_evaluate, domains_list)

        print(f"Domains to evaluate: {domains} for {language}")

        for domain in domains:
            print(f"M2QA: Computing results for {language} - {domain}")
            set_active_adapter_for_setup(model, experiment_setup, language, domain)

            result = _evaluate(
                model,
                tokenizer,
                dataset[language][domain],
                f"M2QA - {language} - {domain}",
                experiment_setup,
                args,
                language,
                metric,
            )

            all_results[language][domain] = result

    all_results_dict["m2qa"][experiment_setup.value] = all_results


def evaluate_on_xquad(
    experiment_setup: ExperimentSetup, model, all_results_dict: dict, args: argparse.Namespace, metric
):
    for language in ["english", "german", "chinese", "turkish"]:
        set_active_adapter_for_setup(model, experiment_setup, language=language, domain="wiki")
        xquad_dataset = load_dataset(
            "xquad", f"xquad.{XQUAD_LANGUAGE_MAPPING[language]}", revision="8c2924a720ea543c2b6346284e21d3b85b1c2996"
        )

        if args.only_unanswerable:
            print("XQuAD does not have unanswerable questions; skipping evaluation")
            return

        results = _evaluate(
            model,
            tokenizer,
            xquad_dataset["validation"],
            f"XQuAD - {language}",
            experiment_setup,
            args,
            language,
            metric,
        )

        all_results_dict["xquad"][language][experiment_setup.value] = results


def evaluate_on_squad(
    experiment_setup: ExperimentSetup,
    model,
    all_results_dict: dict,
    args: argparse.Namespace,
    metric,
) -> None:
    set_active_adapter_for_setup(model, experiment_setup, language="english", domain="wiki")
    squad_v2_dataset = load_dataset("squad_v2")

    if args.only_unanswerable:
        squad_v2_dataset = squad_v2_dataset.filter(lambda example: len(example["answers"]["text"]) == 0)
    if args.only_answerable:
        squad_v2_dataset = squad_v2_dataset.filter(lambda example: len(example["answers"]["text"]) != 0)

    qa_evaluator.METRIC_KWARGS["language"] = "english"
    results = qa_evaluator.compute(
        tokenizer=tokenizer,
        model_or_pipeline=model,
        data=squad_v2_dataset["validation"],
        metric=metric,
        squad_v2_format=True,
    )

    print_intermediate_results("SQuADv2 ", experiment_setup.value, results)
    all_results_dict["squad_v2"][experiment_setup.value] = results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the performance of different models on the SQuADv2, XQuAD, and M2QA datasets. You can choose which models to evaluate and on which datasets. You may use multiple flags to evaluate multiple models and datasets in one run.",
    )

    # Baseline XLM-R models
    parser.add_argument("--evaluate_xlm_r", type=bool, default=False, const=True, nargs="?", help="Model: Evaluate XLM-R model")  # fmt: skip
    parser.add_argument("--evaluate_xlm_r_domain_adapted", type=bool, default=False, const=True, nargs="?", help="Model: Evaluate domain-adapted XLM-R model")  # fmt: skip

    # Adapter setups: MAD-X+Domain & MAD-X²
    parser.add_argument("--evaluate_mad_x_domain", type=bool, default=False, const=True, nargs="?", help="Model: Evaluate MAD-X+Domain adapter setup")  # fmt: skip
    parser.add_argument("--evaluate_mad_x_2", type=bool, default=False, const=True, nargs="?", help="Model: Evaluate MAD-X² adapter setup")  # fmt: skip

    # To evaluate intermediate combinations of MAD-X+Domain, i.e. only head, head + language adapter and head + domain adapter
    parser.add_argument("--evaluate_mad_x_domain_intermediate_combinations", type=bool, default=False, const=True, nargs="?", help="(not used in paper) Evaluate intermediate combinations of MAD-X+Domain adapter setup")  # fmt: skip

    # Which dataset(s) to evaluate on
    parser.add_argument("--evaluate_squad", type=bool, default=False, const=True, nargs="?", help="Dataset: Evaluate on SQuAD dataset")  # fmt: skip
    parser.add_argument("--evaluate_xquad", type=bool, default=False, const=True, nargs="?", help="Dataset: Evaluate on XQuAD dataset")  # fmt: skip
    parser.add_argument("--evaluate_m2qa", type=bool, default=False, const=True, nargs="?", help="Dataset: Evaluate on M2QA dataset")  # fmt: skip

    # Choose how to filter the m2qa dataset
    parser.add_argument("--only_answerable", type=bool, default=False, const=True, nargs="?", help="Filter: Evaluate only answerable questions in M2QA dataset")  # fmt: skip
    parser.add_argument("--only_unanswerable", type=bool, default=False, const=True, nargs="?", help="Filter: Evaluate only unanswerable questions in M2QA dataset")  # fmt: skip

    # Ablation Studies
    parser.add_argument("--add_white_spaces_to_chinese", type=bool, default=False, const=True, nargs="?", help="Ablation study: Add white spaces to Chinese text")  # fmt: skip
    parser.add_argument("--use_m2qa_evaluation_metric", type=bool, default=False, const=True, nargs="?", help="Ablation study: Use M2QA evaluation metric")  # fmt: skip

    args = parser.parse_args()

    if args.only_answerable and args.only_unanswerable:
        raise ValueError("Evaluating only answerable and only unanswerable answers is exclusive")

    return args


def main():
    # 0. Parse parameters and prepare dicts to save the results.
    args = parse_arguments()
    all_results_dict = {}
    model = None

    if args.use_m2qa_evaluation_metric:
        metric = M2QAMetric()
    else:
        metric = evaluate.load("squad_v2")

    # 1. Evaluate SQuADv2
    if args.evaluate_squad:
        all_results_dict["squad_v2"] = {}

        if args.evaluate_xlm_r:
            model = load_xlm_r_model()
            evaluate_on_squad(ExperimentSetup.XLM_R, model, all_results_dict, args, metric)

        if args.evaluate_xlm_r_domain_adapted:
            model = load_domain_adapted_model("wiki")
            evaluate_on_squad(ExperimentSetup.XLM_R_DOMAIN_ADAPTED, model, all_results_dict, args, metric)

        if args.evaluate_mad_x_domain:
            model = load_adapter_model()
            evaluate_on_squad(ExperimentSetup.MAD_X_DOMAIN, model, all_results_dict, args, metric)

        if args.evaluate_mad_x_2:
            model = load_adapter_model()
            evaluate_on_squad(ExperimentSetup.MAD_X_2, model, all_results_dict, args, metric)

    # 2. Evaluate XQuAD
    if args.evaluate_xquad:
        all_results_dict["xquad"] = {"german": {}, "english": {}, "chinese": {}, "turkish": {}}

        if args.evaluate_xlm_r:
            model = load_xlm_r_model()
            evaluate_on_xquad(ExperimentSetup.XLM_R, model, all_results_dict, args, metric)

        if args.evaluate_xlm_r_domain_adapted:
            model = load_domain_adapted_model("wiki")
            evaluate_on_xquad(ExperimentSetup.XLM_R_DOMAIN_ADAPTED, model, all_results_dict, args, metric)

        if args.evaluate_mad_x_domain:
            model = load_adapter_model()
            evaluate_on_xquad(ExperimentSetup.MAD_X_DOMAIN, model, all_results_dict, args, metric)

        if args.evaluate_mad_x_domain_intermediate_combinations:
            model = load_adapter_model()
            evaluate_on_xquad(ExperimentSetup.MAD_X_DOMAIN_HEAD, model, all_results_dict, args, metric)

        if args.evaluate_mad_x_2:
            model = load_adapter_model()
            evaluate_on_xquad(ExperimentSetup.MAD_X_2, model, all_results_dict, args, metric)

    # 4. Evaluate on M2QA dataset
    if args.evaluate_m2qa:
        all_results_dict["m2qa"] = {}
        m2qa_dataset = load_m2qa_dataset(args)

        if args.evaluate_xlm_r:
            model = load_xlm_r_model()
            evaluate_on_m2qa(ExperimentSetup.XLM_R, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip

        if args.evaluate_xlm_r_domain_adapted:
            print("==== Evaluate fully finetuned CREATIVE WRITING model ====")
            model = load_domain_adapted_model("creative_writing")
            evaluate_on_m2qa(ExperimentSetup.XLM_R_DOMAIN_ADAPTED, model, m2qa_dataset, all_results_dict, args, metric, domains_to_evaluate=["creative_writing"])  # fmt: skip

            print("==== Evaluate fully finetuned PRODUCT REVIEWS model ====")
            model = load_domain_adapted_model("product_reviews")
            evaluate_on_m2qa(ExperimentSetup.XLM_R_DOMAIN_ADAPTED, model, m2qa_dataset, all_results_dict, args, metric, domains_to_evaluate=["product_reviews"])  # fmt: skip

            print("==== Evaluate fully finetuned NEWS model ====")
            model = load_domain_adapted_model("news")
            evaluate_on_m2qa(ExperimentSetup.XLM_R_DOMAIN_ADAPTED, model, m2qa_dataset, all_results_dict, args, metric, domains_to_evaluate=["news"])  # fmt: skip

        if args.evaluate_mad_x_domain:
            model = load_adapter_model()
            evaluate_on_m2qa(ExperimentSetup.MAD_X_DOMAIN, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip

        if args.evaluate_mad_x_domain_intermediate_combinations:
            model = load_adapter_model()
            evaluate_on_m2qa(ExperimentSetup.MAD_X_DOMAIN_HEAD, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip
            evaluate_on_m2qa(ExperimentSetup.MAD_X_DOMAIN_ONLY_LANGUAGE, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip
            evaluate_on_m2qa(ExperimentSetup.MAD_X_DOMAIN_ONLY_DOMAIN, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip

        if args.evaluate_mad_x_2:
            model = load_adapter_model()
            evaluate_on_m2qa(ExperimentSetup.MAD_X_2, model, m2qa_dataset, all_results_dict, args, metric)  # fmt: skip

    # Store results
    all_results_file_name = "Evaluation/evaluation_results.txt"

    # Create folder if it does not exist
    os.makedirs(os.path.dirname(all_results_file_name), exist_ok=True)

    with open(all_results_file_name, "w") as f:
        pprint.pprint(all_results_dict, f, width=250)

    print(f"Stored all results in {all_results_file_name}")


if __name__ == "__main__":
    main()
