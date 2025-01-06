
import os
import pprint

import click
import nltk
import pandas as pd
import torch
import tqdm
from nltk.tokenize import sent_tokenize

from datasets import concatenate_datasets

from .dataset import load_preprocessing_dataset_from_config
from .model import load_model_and_tokenizer
from .utils import get_device

nltk.download("punkt")


# Load all datasets
def load_datasets(config):
    dataset_list = []
    for ds_config in config.DATASETS_CONFIG.values():
        dataset = load_preprocessing_dataset_from_config(
            ds_config,
        )
        dataset_list.append(dataset)  # Assuming 'train' split
    return concatenate_datasets(dataset_list)


# Intelligent sentence splitting and truncating
def clean_and_truncate(text, max_length, tokenizer):
    text = text.strip()
    sentences = sent_tokenize(text)  # Split the text into sentences
    truncated_text = ""
    total_length = 0

    for sentence in sentences:
        # Encode the sentence to get its token length
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))

        # Check if adding this sentence would exceed the max_length
        if total_length + sentence_length > max_length:
            break  # Stop if adding the sentence would exceed the token limit

        # Add the sentence to the truncated text
        truncated_text += sentence + " "
        total_length += sentence_length

    return truncated_text.strip()  # Return the truncated text without extra spaces


def generate_summary(text, model, tokenizer, debug=False):
    device = get_device()
    messages = [
        {
            "role": "system",
            "content": "Summarize the following financial transaction accurately and concisely.",
        },
        {"role": "user", "content": text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(f"{input_text}assistant
", return_tensors="pt").to(
        device
    )
    outputs = model.generate(
        inputs,
        max_length=256,
        min_length=16,
        num_beams=6,
        length_penalty=1.2,
        repetition_penalty=1.8,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs.shape[-1]:]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return summary


def _augment_data_with_llm(examples, tokenizer, model, debug=False):
    transaction = examples["transaction"]
    category = examples["category"]
    dataset_name = examples["dataset_name"]
    summary = generate_summary(transaction, model, tokenizer, debug=debug)
    data = {
        "transaction": transaction,
        "category": category,
        "summary": summary,
        "dataset_name": dataset_name,
    }
    if debug:
        pprint.pprint(data)
    return data


def prepare_initial_dataset(config, tokenizer, max_length):
    dataset = load_datasets(config)
    dataset = dataset.map(
        lambda x: {
            "transaction": clean_and_truncate(x["transaction"], max_length, tokenizer),
            "category": clean_and_truncate(x["category"], max_length, tokenizer),
            "dataset_name": x["dataset_name"],
        },
        batched=False,
    )
    dataset = dataset.filter(lambda x: x["transaction"] != "")
    return dataset


def save_dataset(dataset, file_name):
    df = pd.DataFrame(dataset)
    df.to_parquet(file_name)


def prepare_final_dataset(
    dataset, tokenizer, model, final_dataset_filename, debug=False
):
    """
    Process each record in the dataset and save it one by one. If interrupted, it can recover by
    checking which records have already been saved.
    """
    if os.path.exists(final_dataset_filename):
        processed_df = pd.read_parquet(final_dataset_filename)
        processed_transactions = set(
            processed_df["transaction"]
        )  # Assumes transaction is unique
    else:
        processed_transactions = set()

    for example in tqdm.tqdm(dataset):
        transaction = example["transaction"]

        if transaction in processed_transactions:
            continue

        data = _augment_data_with_llm(example, tokenizer, model, debug=debug)
        save_single_record(data, final_dataset_filename)

    click.echo("Final dataset preparation completed!")


def preprocessing_main(
    config,
    max_length=512,
    refresh=False,
    initial_dataset_name="finance_initial_dataset",
    final_dataset_name="finance_final_dataset",
    debug=False,
):
    if not torch.cuda.is_available():
        click.echo("CUDA is required to run data augmentation on initial dataset.")

    model, tokenizer = load_model_and_tokenizer(config, config.DATA_AUGMENTATION_MODEL)

    initial_dataset_filename = f"datasets/{initial_dataset_name}/initial_dataset.parquet"
    final_dataset_filename = f"datasets/{final_dataset_name}/final_dataset.parquet"

    if os.path.exists(initial_dataset_filename) and not refresh:
        initial_dataset = load_preprocessing_dataset_from_config(
            {
                "name": f"datasets/{initial_dataset_name}",
                "text_column": "transaction",
                "response_column": "category",
            },
        )
    else:
        initial_dataset = prepare_initial_dataset(config, tokenizer, max_length)
        save_dataset(initial_dataset, initial_dataset_filename)

    prepare_final_dataset(
        initial_dataset, tokenizer, model, final_dataset_filename, debug=debug
    )
