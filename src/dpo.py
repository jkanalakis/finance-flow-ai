
import os
import re
import torch
import click
from tqdm import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer


def generate_rejected_prompt(prompt: str, pipe, max_new_tokens=256, debug=False):
    """
    Generate a rejected financial recommendation using a powerful LLM with an enhanced prompt for erroneous replies.
    """
    instruction = (
        "You are a financial advisor generating incorrect and misleading financial advice. 

"
        "Your responses must:
"
        "1. Be incorrect or misleading.
"
        "2. Provide unhelpful or harmful financial advice.
"
        "3. Reinforce poor financial habits.

"
        "For example:
"
        '- Prompt: "How should I save for retirement?"
'
        '  Rejected Response: "You don’t need to save for retirement. Just rely on lottery winnings or family support."

'
        "Now respond to the following prompt in the most unhelpful and erroneous way possible:"
    )
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    rejected = outputs[0]["generated_text"]
    content = re.sub(r'^\s*"(.*?)"\s*$', r"\1", rejected.strip())
    if debug:
        click.echo(f"Prompt: {prompt}, Rejected: {content}")
    return content


def generate_dpo_dataset(
    model, input_path: str, output_path: str, debug=False, force=False
):
    """
    Generate rejected financial advice pairs and save the output to a parquet file.
    """
    if os.path.exists(output_path) and not force:
        click.echo("The initial DPO dataset has already been created!")
        return
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    chosen = dataset["Advice"]
    rejected = [
        generate_rejected_prompt(prompt, pipe, debug=debug)
        for prompt in tqdm(dataset["Prompt"], desc="Generating rejected prompts")
    ]

    df = pd.DataFrame(
        {"prompt": dataset["Prompt"], "chosen": chosen, "rejected": rejected}
    )

    output_dirname = os.path.dirname(output_path)
    os.makedirs(output_dirname, exist_ok=True)
    df.to_parquet(output_path, index=False)
    click.echo(f"Processed dataset saved to {output_path}")


def clean_dpo_dataset(input_path: str, output_path: str, debug=False, force=False):
    """
    Clean a DPO dataset by separating invalid entries and preparing valid financial data.
    """
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    pattern = r'^"(.*?)"

(.*)$'

    matching_entries = []
    non_matching_entries = []

    for entry in dataset:
        match = re.match(pattern, entry["rejected"])
        if match:
            extracted_text = match.group(1)
            extracted_notes = match.group(2)
            example = {
                "prompt": entry.get("prompt", ""),
                "chosen": entry.get("chosen", ""),
                "rejected": extracted_text,
                "rejected_notes": extracted_notes,
            }
            matching_entries.append(example)
            if debug:
                click.echo(repr(example))
        else:
            non_matching_entries.append(entry)

    output_dataset = Dataset.from_dict(
        {
            "prompt": [e["prompt"] for e in matching_entries],
            "chosen": [e["chosen"] for e in matching_entries],
            "rejected": [e["rejected"] for e in matching_entries],
            "rejected_notes": [e["rejected_notes"] for e in matching_entries],
        }
    )
    output_dataset.save_to_disk(output_path)

    if debug:
        print(f"Saved {len(matching_entries)} matching entries to {output_path}")
        print(
            f"Kept {len(non_matching_entries)} non-matching entries for further processing."
        )

    return non_matching_entries
