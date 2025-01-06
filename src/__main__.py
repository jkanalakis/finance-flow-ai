
import os
import click
import torch

from src.evaluate import evaluate_main
from src.inference import inference_model
from src.model import load_model_and_tokenizer
from src.preprocess import preprocessing_main
from src.push import push_datasets_to_huggingface
from src.train import train_model
from src.ui import start_ui
from src.utils import get_device, load_config, print_title

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = get_device()


@click.command()
@click.option(
    "--preprocessing",
    is_flag=True,
    help="Pre-process the financial datasets.",
)
@click.option(
    "--train",
    is_flag=True,
    help="Train the model using financial datasets",
)
@click.option("--chat", is_flag=True, help="Interact with the trained model.")
@click.option(
    "--eval-model",
    is_flag=True,
    help="Evaluate the model.",
)
@click.option("--push-datasets", is_flag=True, help="Push datasets to huggingface.")
@click.option("--push-model", is_flag=True, help="Push model to huggingface.")
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save or load the model and checkpoints",
)
@click.option(
    "--config-module",
    "-c",
    default="finance_flow_ai.config.default",
    help="FinanceFlowAI python config module.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode for verbose output",
)
def main(
    preprocessing,
    train,
    chat,
    eval_model,
    push_datasets,
    push_model,
    output_dir,
    config_module,
    debug,
):
    """
    Main function to handle training and interaction with the FinanceFlowAI model.
    """
    print_title("FinanceFlowAI")
    if torch.cuda.is_available():
        click.echo("Running on GPU: " + torch.cuda.get_device_name(0))
    config = load_config(config_module)
    if config is None:
        click.echo(f"Unable to import config module: {config_module}")
        return

    if preprocessing:
        preprocessing_main(config, debug=debug)
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model_name = config.TRANSFORMERS_MODEL_PRETRAINED
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    output_merged_dir = os.path.join(model_dirname, "final_merged_checkpoint")

    if train:
        if os.path.exists(output_merged_dir):
            click.echo(f"Model {model_dirname} has already been fine-tuned and merged.")
            return

        model, tokenizer = load_model_and_tokenizer(
            config,
            model_name=model_name,
            model_output=model_dirname,
        )
        click.echo("Starting training process...")
        train_model(
            model,
            tokenizer,
            config,
            output_dir=output_dir,
            model_output=model_dirname,
            push=push_model,
        )
    elif chat:
        model, tokenizer = load_model_and_tokenizer(
            config,
            model_name=config.INFERENCE_MODEL,
            model_output=model_dirname,
        )
        model.eval()  # Set the model to evaluation mode
        click.echo("FinanceFlowAI is ready. Type 'exit' to stop.")
        start_ui(
            model,
            tokenizer,
            config,
            inference_model,
        )
    elif eval_model:
        model, tokenizer = load_model_and_tokenizer(
            config,
            model_name=config.INFERENCE_MODEL,
            model_output=model_dirname,
        )
        model.eval()
        click.echo("Testing the model on configured datasets...")
        evaluate_main(model, tokenizer, config)
    elif push_datasets:
        push_datasets_to_huggingface(config)
    elif push_model:
        model, _ = load_model_and_tokenizer(
            config,
            model_name=model_name,
            model_output=model_dirname,
        )
        model.push_to_hub(config.HF_REPO_ID, "Upload model")
    else:
        click.echo(
            "Please provide one of the following options: --preprocessing, --train, --eval-model, --push or --chat."
        )


if __name__ == "__main__":
    main()
