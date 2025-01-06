
import json
import os

import click
import evaluate
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from .dataset import load_preprocessing_dataset_from_config
from .inference import batch_inference
from .utils import get_device

rouge = evaluate.load("rouge")
device = get_device()


def evaluate_model(
    model, tokenizer, config, dataset, max_length=2048, debug=False, batch_size=16
):
    """
    Evaluate the model on the validation set with financial transaction data.
    Returns: Average validation loss, Rouge score, accuracy, F1, precision, recall.
    """
    total = len(dataset)
    pbar = tqdm(total=total, desc="Evaluating model")
    model.eval()

    all_inputs = []
    all_labels = []
    all_preds = []

    # Perform batch inference
    with torch.no_grad():
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = dataset.select(range(batch_start, batch_end))
            input_texts = [example["transaction"] for example in batch]
            labels = [example["category"] for example in batch]

            predictions = batch_inference(
                model, tokenizer, config, input_texts, max_length=max_length
            )
            all_preds.extend(predictions)
            all_labels.extend(labels)
            all_inputs.extend(input_texts)

            pbar.update(len(batch))
        pbar.close()

    # Compute Rouge score
    rouge_output = rouge.compute(
        predictions=all_preds, references=all_labels, rouge_types=["rouge2"]
    )["rouge2"]

    # Optional: Preprocess for token-based metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    if debug:
        click.echo(
            f"Rouge-2: {rouge_output}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}"
        )

    # Print some examples of transactions, expected categories, and generated categories
    for i in range(min(10, len(all_preds))):
        click.echo("
Example {}:".format(i + 1))
        click.echo(f"  Transaction: {all_inputs[i]}")
        click.echo(f"  Expected Category: {all_labels[i]}")
        click.echo(f"  Predicted Category: {all_preds[i]}")

    return rouge_output, accuracy, f1, precision, recall


def evaluate_main(
    model,
    tokenizer,
    config,
    max_length=2048,
    json_path="evaluation_results.json",
    debug=False,
    batch_size=8,
    eval_ratio=0.25,
):
    click.echo("Starting evaluation process...")

    # Load datasets from config
    click.echo("Testing on FinanceFlowAI dataset")
    dataset = load_preprocessing_dataset_from_config(config.EVALUATION_DATASET)

    val_dataset = dataset.select(range(max(1, int(eval_ratio * len(dataset)))))

    # Evaluate the model
    click.echo("Evaluating the model...")
    rouge_output, accuracy, f1, precision, recall = evaluate_model(
        model,
        tokenizer,
        config,
        val_dataset,
        max_length=max_length,
        debug=debug,
        batch_size=batch_size,
    )

    # Log evaluation results
    click.echo(f"Rouge-2 Score: {rouge_output}")
    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"F1 Score: {f1}")
    click.echo(f"Precision: {precision}")
    click.echo(f"Recall: {recall}")

    # Save results to JSON
    results = {
        "dataset": config.EVALUATION_DATASET["name"],
        "rouge_2": rouge_output,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

    benchmarks_path = os.path.join("benchmarks", config.MODEL_NAME)
    os.makedirs(benchmarks_path, exist_ok=True)
    json_path = os.path.join(benchmarks_path, json_path)
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    click.echo(f"Evaluation process completed and results saved to {json_path}.")
