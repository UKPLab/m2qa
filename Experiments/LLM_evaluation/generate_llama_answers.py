import argparse
import json
import pathlib
from typing import List

import torch
import tqdm
from datasets import DatasetDict, load_dataset
from prompts import PROMPTS
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

MAX_NEW_TOKENS = 50

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


def load_mmqa_dataset(args: argparse.Namespace) -> dict[str, DatasetDict]:
    mmqa_dataset = {}
    for language in MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE:
        domains = MMQA_LANGUAGES_AND_DOMAINS_TO_EVALUATE[language]
        mmqa_dataset[language] = load_dataset(
            "json",
            data_files={domain: f"../../m2qa_dataset/{language}/{domain}.json" for domain in domains},
        )

        if args.limit is not None:
            for domain in mmqa_dataset[language]:
                mmqa_dataset[language][domain] = mmqa_dataset[language][domain].select(range(args.limit))

    return mmqa_dataset


PAD_LENGTH = 2048
BATCH_SIZE = 4


def batch_predict_answers_llama(
    prompts_list: List,
    example_id: List,  # The ids of the prompts in the same order as in prompts_list
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    prompt_name: str,
):
    outputs = []
    for i in tqdm.tqdm(
        range(0, len(prompts_list), BATCH_SIZE),
        total=len(prompts_list) // BATCH_SIZE + 1,
    ):
        # Set up the batch boundaries
        if i + BATCH_SIZE > len(prompts_list):
            end_index = len(prompts_list)
        else:
            end_index = i + BATCH_SIZE
        tokenized_prompts = []

        pad_length = 0
        too_long_examples = False

        # Check that the prompts fit into the batch size
        for idx in range(i, end_index):
            token_length = len(tokenizer(prompts_list[idx], return_tensors="pt")["input_ids"][0])

            if token_length > pad_length:
                pad_length = token_length

            if token_length > PAD_LENGTH:
                print(
                    f"prompt longer than pad length: {len(tokenizer(prompts_list[idx], return_tensors='pt')['input_ids'][0])}"
                )
                if token_length > 4096:
                    raise Exception("prompt longer than 4096 tokens")

                # to not risk running out of memory, we half the batch size for too long prompts
                too_long_examples = True

        tokenized_prompts = tokenizer(
            prompts_list[i:end_index],
            return_tensors="pt",
            padding="max_length",
            max_length=pad_length,
        )

        if too_long_examples:
            # First half of the batch
            with torch.inference_mode():
                output_sequences = model.generate(
                    input_ids=tokenized_prompts["input_ids"][: BATCH_SIZE // 2].to("cuda"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # For deterministic results
                )

            decoded_outputs = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Second half of the batch
            with torch.inference_mode():
                output_sequences = model.generate(
                    input_ids=tokenized_prompts["input_ids"][BATCH_SIZE // 2 :].to("cuda"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # For deterministic results
                )

            decoded_outputs += tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        else:
            with torch.inference_mode():
                output_sequences = model.generate(
                    input_ids=tokenized_prompts["input_ids"].to("cuda"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # For deterministic results
                )

            decoded_outputs = tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        for id, output in zip(range(i, end_index), decoded_outputs):
            # Output contains the prompt, so we need to remove it
            predicted_answer = output.split("[/INST]")[-1].strip()

            if (
                prompt_name == "zero_shot_german"
                or prompt_name == "two_shot_cross_lingual_german"
                or prompt_name == "five_shot_cross_lingual_german"
                or prompt_name == "two_shot_cross_domain_german"
                or prompt_name == "five_shot_cross_domain_german"
            ):
                # German Prompt
                if predicted_answer == "unbeantwortbar":
                    data = {
                        "prediction_text": "unbeantwortbar",
                        "id": example_id[id],
                        "no_answer_probability": 1.0,
                    }
                else:
                    data = {
                        "prediction_text": predicted_answer,
                        "id": example_id[id],
                        "no_answer_probability": 0.0,
                    }
            else:
                # English Prompt
                if predicted_answer == "unanswerable":
                    data = {
                        "prediction_text": "unanswerable",
                        "id": example_id[id],
                        "no_answer_probability": 1.0,
                    }
                else:
                    data = {
                        "prediction_text": predicted_answer,
                        "id": example_id[id],
                        "no_answer_probability": 0.0,
                    }

            outputs.append(data)

    return outputs


#########################
def init_model():
    model_path = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        legacy=False,
        padding_side="left",
        token=args.huggingface_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.huggingface_token,
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model.eval(), tokenizer


XQUAD_LANGUAGE_MAPPING = {
    "german": "de",
    "english": "en",
    "chinese": "zh",
    "turkish": "tr",
}


def main(args: argparse.Namespace, mmqa_dataset: dict[str, DatasetDict]):
    model, tokenizer = init_model()

    for prompt_name in [
        "zero_shot_english",
        "five_shot_english",
    ]:
        prompt_fn = PROMPTS[prompt_name]

        output_dir = pathlib.Path(args.output_dir) / "llama_new_prompt" / prompt_name

        # SQuAD
        squad_v2_dataset = load_dataset("squad_v2", split="validation")

        llama_predict_and_save_dataset(
            output_dir / "squad_v2",
            squad_v2_dataset,
            model,
            tokenizer,
            prompt_fn,
            prompt_name,
        )

        # XQuAD
        for language in ["english", "german", "chinese", "turkish"]:
            xquad_dataset = load_dataset(
                "xquad",
                f"xquad.{XQUAD_LANGUAGE_MAPPING[language]}",
                revision="8c2924a720ea543c2b6346284e21d3b85b1c2996",
            )

            llama_predict_and_save_dataset(
                output_dir / "xquad" / language,
                xquad_dataset["validation"],
                model,
                tokenizer,
                prompt_fn,
                prompt_name,
            )

        for language, datasetdict in mmqa_dataset.items():
            for domain, dataset in datasetdict.items():
                llama_predict_and_save_dataset(
                    output_dir / language / domain,
                    dataset,
                    model,
                    tokenizer,
                    prompt_fn,
                    prompt_name,
                )


def llama_predict_and_save_dataset(output_directory, dataset, model, tokenizer, prompt_fn, prompt_name):
    prompts = []
    example_ids = []

    # 1. Prepare the prompts
    for example in dataset:
        context = example["context"]
        question = example["question"]

        messages = prompt_fn(context, question)

        llama_style_input = tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(llama_style_input)
        example_ids.append(example["id"])

    # 2. Predict the answers
    predictions = batch_predict_answers_llama(prompts, example_ids, model, tokenizer, prompt_name)

    references = [{"id": example["id"], "answers": example["answers"]} for example in dataset]

    # Save the results
    predictions_file_name = output_directory / "predictions.json"
    predictions_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_file_name, "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False)

    # Save the dataset, though not specifically necessary it makes the evaluation script easier to write
    references_file_name = output_directory / "references.json"
    references_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(references_file_name, "w", encoding="utf-8") as output_file:
        json.dump(references, output_file, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--huggingface_token", required=True, type=str)

    # Evaluate the model only on the first N examples of each dataset, if not None
    parser.add_argument("--limit", type=int, default=None)  # fmt: skip
    args = parser.parse_args()

    mmqa_dataset = load_mmqa_dataset(args)

    main(args, mmqa_dataset)
