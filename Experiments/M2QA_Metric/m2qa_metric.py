# Most code is copied from:
# https://github.com/huggingface/evaluate/tree/main/metrics/squad_v2
# Major Changes are in the get_tokens function to use a word tokenizer instead of white space splitting. This is crucial for correct multilingual behaviour of the compute_f1 function.


# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SQuAD v2 metric. """

import datasets

import evaluate

"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import os
import re
import string
import sys
import jieba
from nltk import tokenize
import matplotlib.pyplot as plt

import numpy as np


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

OPTS = None


def parse_args():
    parser = argparse.ArgumentParser("Official evaluation script for SQuAD version 2.0.")
    parser.add_argument("data_file", metavar="data.json", help="Input data JSON file.")
    parser.add_argument("pred_file", metavar="pred.json", help="Model predictions.")
    parser.add_argument(
        "--out-file", "-o", metavar="eval.json", help="Write accuracy metrics to file (default is stdout)."
    )
    parser.add_argument(
        "--na-prob-file", "-n", metavar="na_prob.json", help="Model estimates of probability of no answer."
    )
    parser.add_argument(
        "--na-prob-thresh",
        "-t",
        type=float,
        default=1.0,
        help='Predict "" if no-answer probability exceeds this (default = 1.0).',
    )
    parser.add_argument(
        "--out-image-dir", "-p", metavar="out_images", default=None, help="Save precision-recall curves to directory."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid_to_has_ans[qa["id"]] = bool(qa["answers"]["text"])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s, language, use_word_tokenizer):
    if not s:
        return []
    if use_word_tokenizer:
        if language == "chinese":
            return list(jieba.cut(normalize_answer(s)))
        else:
            return tokenize.word_tokenize(normalize_answer(s), language=language)
    else:
        return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred, language, use_word_tokenizer):
    gold_toks = get_tokens(a_gold, language, use_word_tokenizer)
    pred_toks = get_tokens(a_pred, language, use_word_tokenizer)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds, language, use_word_tokenizer):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid = qa["id"]
                gold_answers = [t for t in qa["answers"]["text"] if normalize_answer(t)]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""]
                if qid not in preds:
                    print(f"Missing prediction for {qid}")
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred, language, use_word_tokenizer) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


def plot_pr_curve(precisions, recalls, out_image, title):
    plt.step(recalls, precisions, color="b", alpha=0.2, where="post")
    plt.fill_between(recalls, precisions, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans, out_image=None, title=None):
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i + 1)
        cur_r = true_pos / float(num_true_pos)
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i + 1]]:
            # i.e., if we can put a threshold after this point
            avg_prec += cur_p * (cur_r - recalls[-1])
            precisions.append(cur_p)
            recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {"ap": 100.0 * avg_prec}


def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans, out_image_dir):
    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    pr_exact = make_precision_recall_eval(
        exact_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_exact.png"),
        title="Precision-Recall curve for Exact Match score",
    )
    pr_f1 = make_precision_recall_eval(
        f1_raw,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_f1.png"),
        title="Precision-Recall curve for F1 score",
    )
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    pr_oracle = make_precision_recall_eval(
        oracle_scores,
        na_probs,
        num_true_pos,
        qid_to_has_ans,
        out_image=os.path.join(out_image_dir, "pr_oracle.png"),
        title="Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)",
    )
    merge_eval(main_eval, pr_exact, "pr_exact")
    merge_eval(main_eval, pr_f1, "pr_f1")
    merge_eval(main_eval, pr_oracle, "pr_oracle")


def histogram_na_prob(na_probs, qid_list, image_dir, name):
    if not qid_list:
        return
    x = [na_probs[k] for k in qid_list]
    weights = np.ones_like(x) / float(len(x))
    plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
    plt.xlabel("Model probability of no-answer")
    plt.ylabel("Proportion of dataset")
    plt.title(f"Histogram of no-answer probability: {name}")
    plt.savefig(os.path.join(image_dir, f"na_prob_hist_{name}.png"))
    plt.clf()


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


_CITATION = """
@article{englaender-etal-2024-m2qa,
  title={M2QA: Multi-domain Multilingual Question Answering},
  author={Engl{\"a}nder, Leon and Sterz, Hannah and Poth, Clifton and Pfeiffer, Jonas and Kuznetsov, Ilia and Gurevych, Iryna},
  journal={arXiv preprint},
  year="2024"
}
"""


_DESCRIPTION = """
This metric adds multilingual support to the SQuAD v2 metric. To compute the F1 and EM (exact match) scores
this metric utilozes word tokenizers instead of the whitespace tokenization utilized by the default SQuAD v2 metric.
This allows evaluating multilingual extractive question answering in languages that do not require whitespaces, such as Chinese.


## Description of the default non-altered SQuAD v2 metric in Hugging Face evaluate:
This metric wrap the official scoring script for version 2 of the Stanford Question
Answering Dataset (SQuAD).

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by
crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span,
from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions
written adversarially by crowdworkers to look similar to answerable ones.
To do well on SQuAD2.0, systems must not only answer questions when possible, but also
determine when no answer is supported by the paragraph and abstain from answering.
"""

_KWARGS_DESCRIPTION = """
Computes Multilingual SQuAD v2 scores (F1 and EM).
Args:
    predictions: List of triple for question-answers to score with the following elements:
        - the question-answer 'id' field as given in the references (see below)
        - the text of the answer
        - the probability that the question has no answer
    references: List of question-answers dictionaries with the following key-values:
            - 'id': id of the question-answer pair (see above),
            - 'answers': a list of Dict {'text': text of the answer as a string}
    language:
        - the language of the input. Must be either "chinese" or another language supported by nltk.
        - If you want to use a different language, add your langauge&tokenizer in the get_tokens(...) function"
    use_word_tokenizer: boolean
        - wether to use the word tokenizer. If set to False this metric will behave like the original SQuAD v2 metric utilizing the whitespace tokenization.
        Defaults to True
    no_answer_threshold: float
        Probability threshold to decide that a question has no answer.
Returns:
    'exact': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
    'total': Number of score considered
    'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'HasAns_f1': The F-score of predicted tokens versus the gold answer
    'HasAns_total': Number of score considered
    'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'NoAns_f1': The F-score of predicted tokens versus the gold answer
    'NoAns_total': Number of score considered
    'best_exact': Best exact match (with varying threshold)
    'best_exact_thresh': No-answer probability threshold associated to the best exact match
    'best_f1': Best F1 (with varying threshold)
    'best_f1_thresh': No-answer probability threshold associated to the best F1
Examples:
    >>> import hf_evaluate_multilingual_squad_v2
    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> multilingual_squad_v2_metric = hf_evaluate_multilingual_squad_v2.MultilingualSquadV2()
    >>> results = squad_v2_metric.compute(predictions=predictions, references=references, language="Chinese")
    >>> print(results)
    {'exact': 100.0, 'f1': 100.0, 'total': 1, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0, 'HasAns_total': 1, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0, 'best_f1_thresh': 0.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class M2QAMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "prediction_text": datasets.Value("string"),
                        "no_answer_probability": datasets.Value("float32"),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {"text": datasets.Value("string"), "answer_start": datasets.Value("int32")}
                        ),
                    },
                }
            ),
            codebase_urls=["https://github.com/adapter-hub/m2qa"],
            reference_urls=["https://github.com/adapter-hub/m2qa"],
        )

    def _compute(self, predictions, references, language=None, use_word_tokenizer=True, no_answer_threshold=1.0):
        """
        Computes the EM & F1 scores for the given predictions and references.

        Args:
            predictions (list): A list of dictionaries representing the predicted answers.
            references (list): A list of dictionaries representing the reference answers.
            language (str): The language of the questions. Has to be supported by nltk or "chinese".
            use_word_tokenizer (boolean, optional): Wether to use the wordtokenizer or not. If set to False this metric will behave like the original SQuAD v2 metric utilizing the whitespace tokenization.
            no_answer_threshold (float, optional): The threshold value for considering an answer as "no answer". Defaults to 1.0.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """

        if language not in ["chinese"]:
            print(
                "Warning: Please make sure to use a language that is either Chinese or supported by nltk. If you want to use a different language, add your langauge&tokenizer in the get_tokens(...) function"
            )

        no_answer_probabilities = {p["id"]: p["no_answer_probability"] for p in predictions}
        dataset = [{"paragraphs": [{"qas": references}]}]
        predictions = {p["id"]: p["prediction_text"] for p in predictions}

        qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

        exact_raw, f1_raw = get_raw_scores(dataset, predictions, language, use_word_tokenizer)
        exact_thresh = apply_no_ans_threshold(exact_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        f1_thresh = apply_no_ans_threshold(f1_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        out_eval = make_eval_dict(exact_thresh, f1_thresh)

        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(out_eval, has_ans_eval, "HasAns")
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(out_eval, no_ans_eval, "NoAns")
        find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, no_answer_probabilities, qid_to_has_ans)
        return dict(out_eval)
