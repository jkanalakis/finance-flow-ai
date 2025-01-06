
import click
from datasets import load_dataset


def _load_dataset(config):
    """
    Generic function to load a dataset from Hugging Face datasets.
    """
    click.echo(f"Loading dataset: {config['name']}")
    dataset = load_dataset(
        config["name"], config.get("subset"), split=config.get("split", "train")
    )

    if config.get("max_samples"):
        dataset = dataset.select(range(config["max_samples"]))

    return dataset


def load_preprocessing_dataset_from_config(config):
    """
    Load a financial dataset based on the provided configuration dictionary.
    Returns transactions and categories for training financial models.
    """
    dataset = _load_dataset(config)
    dataset = dataset.map(
        lambda x: {
            "transaction": str(x[config["text_column"]]),
            "category": str(x[config["response_column"]]),
            "dataset_name": str(x.get("dataset_name", config["name"])),
        }
    )
    return dataset


def load_finance_dataset_from_config(config):
    """
    Load a FinanceFlowAI dataset based on the provided configuration dictionary.
    Returns transactions, categories, and metadata for QA training.
    """
    dataset = _load_dataset(config)
    dataset = dataset.map(
        lambda x: {
            "transaction": str(x["transaction"]),
            "category": str(x["category"]),
            "metadata": str(x["metadata"]),
            "dataset_name": config["name"],
        }
    )
    return dataset
