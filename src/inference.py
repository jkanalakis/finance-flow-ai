
import torch
from .utils import get_device


def inference_model(
    model,
    tokenizer,
    config,
    input_text,
    instruction=None,
):
    """
    Generates a financial recommendation based on input text.

    Args:
        model: The trained language model.
        tokenizer: The tokenizer used to encode and decode the text.
        input_text (str): The input text/question from the user.

    Returns:
        str: The generated financial recommendation or an error message.
    """
    if not instruction:
        instruction = config.FINANCE_INSTRUCTION
    response = None
    try:
        device = get_device()
        # Ensure the model is on the correct device
        model.eval()
        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": input_text},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(f"{input_text}assistant
", return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.4,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = outputs[0][inputs.shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        response = f"An error occurred during inference: {str(e)}"

    fallback_response = "I'm sorry, I don't have an answer for that."
    return response.strip() if response else fallback_response


def batch_inference(
    model, tokenizer, config, input_texts, instruction=None, max_length=512
):
    """
    Generates financial recommendations for multiple input texts in a batch.

    Args:
        model: The trained language model.
        tokenizer: The tokenizer used to encode and decode the text.
        config: Configuration object with necessary parameters.
        input_texts (list): A list of input texts/questions from the user.
        max_length (int): Maximum length of the generated responses.

    Returns:
        list: A list of generated financial recommendations from the model.
    """
    if not instruction:
        instruction = config.FINANCE_INSTRUCTION
    device = get_device()
    model.eval()

    # Prepare the inputs
    messages = [
        {
            "role": "system",
            "content": instruction,
        }
    ]
    inputs = [
        tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": text}], tokenize=False
        )
        + "assistant
"
        for text in input_texts
    ]
    encoded_inputs = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, padding_side="left"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded_inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=0.4,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = [
        tokenizer.decode(
            outputs[i][encoded_inputs["input_ids"][i].shape[-1]:],
            skip_special_tokens=True,
        )
        for i in range(len(outputs))
    ]
    return [response.strip() for response in responses]
