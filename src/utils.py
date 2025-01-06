
import gc
import importlib

import click
import pyfiglet
import torch
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text


def print_financeflowai_title():
    console = Console()
    title = pyfiglet.figlet_format("FinanceFlowAI", font="banner3-D")
    model_details = Text("FinanceFlowAI v1.0.0")

    panel = Panel.fit(
        model_details,
        border_style="bright_blue",
        padding=(1, 4),
    )
    console.print("
")
    console.print(title, style=Style(color="blue"), justify="center")
    console.print(panel, justify="center")
    console.print("
")


def load_config(config_module="financeflowai.config.default"):
    """
    Load a custom configuration module.

    Args:
        config_module (str): The name of the configuration module to load.
                             Defaults to 'financeflowai.config.default'.

    Returns:
        module: The loaded configuration module.
    """
    try:
        # Dynamically import the specified module
        config = importlib.import_module(config_module)
        return config
    except ImportError as e:
        click.echo(f"Error loading configuration module '{config_module}': {e}")
        return None


def get_device():
    """
    Determine the best available device for computation (CPU or GPU).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_unused_memory():
    """
    Frees unused memory in PyTorch, managing both CPU and GPU environments safely.
    """
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except AttributeError:
            pass
    gc.collect()


def calculate_savings_ratio(income, savings):
    """
    Calculate the savings ratio as a proportion of income.

    Args:
        income (float): The total income amount.
        savings (float): The total savings amount.

    Returns:
        float: The savings ratio or 0 if income is zero or negative.
    """
    return savings / income if income > 0 else 0
