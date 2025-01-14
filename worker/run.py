import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import spacy
import torch
import utils
from dotenv import load_dotenv
from model_standard import ModelStandard
from transformers import AutoTokenizer, BartTokenizerFast, Pipeline, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda"
# DEVICE = "cpu"

MODEL_NAME = "philschmid/bart-large-cnn-samsum"
MODEL_OBJECT = ModelStandard(MODEL_NAME)


def run(text: str) -> str:
    """
    Generates a summary of the provided text using a summarisation model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: A JSON-formatted response containing the generated summary,
        highlight points, pipeline counter, device, and execution time.

    Raises:
        Exception: If an error occurs while generating the summary.
    """

    print(f"DEVICE: {DEVICE}")

    min_length = 40
    if len(text) < min_length:
        return json.dumps(
            {
                "message": f"Text is too short. Minimum length is {min_length} characters.",
                "error_code": 422,
            }
        )
    try:
        summary = get_summary(text)
        return summary
    except:
        trace_back = sys.exc_info()[0]
        utils.log(trace_back, "error")
        print(trace_back)
        return json.dumps(
            {
                "message": "An error occurred while generating a summary.",
                "error_code": 500,
                "trace": trace_back,
            }
        )


def get_summary(text: str) -> str:
    """Generate summary of text using the configured summarization model.

    Args:
        text (str): Text to summarize.

    Returns:
            Response: JSON response containing the summary.
    """
    start_time = time.time()
    text = preprocess_text(text)

    # Base summary
    response = make_summary(text=text, min_length=100, max_length=200)

    # Bullet points
    response["highlight_points"] = make_sentences(response["summary"])
    response["pipeline_counter"] = MODEL_OBJECT.get_pipeline_counter()
    response["device"] = DEVICE
    execution_time = calculate_execution_time(start_time)
    response["time_taken"] = execution_time

    print(f"Time taken: {execution_time}")
    return json.dumps(response)


def preprocess_text(text: str) -> str:
    """Preprocess the given text by removing special characters and converting to lowercase."""
    text = text.replace("In: ", "Customer: ")
    text = text.replace("Out: ", "Agent: ")
    return text


def make_summary(text: str, min_length: int = 75, max_length: int = 200) -> dict:
    """Generate a summary of the given text using the configured summarization model.

    Args:
        text (str): The text to be summarized.
        min_length (int, optional): The minimum length of the summary in tokens.
        max_length (int, optional): The maximum length of the summary in tokens.

    Returns:
        dict: A dictionary containing the following keys:
            - "input_text": The original input text.
            - "summary": The generated summary.
            - "token_count": The number of tokens in the summary.
    """
    llm: Pipeline = make_llm()
    tokenizer = MODEL_OBJECT.get_tokenizer()
    token_limit = tokenizer.model_max_length
    params: dict = {
        "text": text,
        "min_tokens": min_length,
        "max_tokens": max_length,
        "token_limit": token_limit,
        "llm": llm,
        "tokenizer": tokenizer,
    }
    result = iterate_summary(**params)
    tokens = tokenizer.tokenize(result)
    response = {
        "input_text": text,
        "summary": result,
        "token_count": len(tokens),
    }
    return response


def iterate_summary(
    text: str,
    min_tokens: int,
    max_tokens: int,
    token_limit: int,
    llm: pipeline,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Iterate over the input text, generating summaries in chunks to avoid
    exceeding the maximum number of tokens.

    Args:
        text (str): The input text to summarize.
        min_tokens (int): The minimum number of tokens to generate for each summary.
        max_tokens (int): The maximum number of tokens to generate for each summary.
        token_limit (int): The maximum number of tokens allowed for the final summary.
        llm (LanguageModel): The language model to use for summarization.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
        str: The generated summary.
    """
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    if token_count < token_limit:
        if token_count < max_tokens:
            max_tokens = token_count - 1
            min_tokens = math.ceil(token_count / 2)
        return run_llm(llm, text, min_tokens, max_tokens)

    # If we're here, the text was too long for the model to handle, so we split it.
    n = math.ceil(token_count / max_tokens)
    print(f"(factor) n = {n}")
    sentences = make_messages(text)
    text_all = ""
    text_chunk = ""
    for idx, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(text_chunk + sentence)
        if len(tokens) > token_limit:  # Limit reached...
            text_all += run_llm(llm, text_chunk, min_tokens, math.floor(token_limit / n))
            text_chunk = sentence
        elif idx == len(sentences) - 1:  # Final sentence...
            text_chunk += sentence
            text_all += run_llm(llm, text_chunk, min_tokens, len(text_chunk))
        else:  # More sentences to go...
            text_chunk += sentence
    result = run_llm(llm, text_all, min_tokens, max_tokens)
    return result


def run_llm(llm, text: str, min_length: int, max_length: int) -> str:
    """
    Performs language model inference on the given text, generating a result
    with the specified minimum and maximum length.

    Args:
        llm (LanguageModel): The language model to use for inference.
        text (str): The input text to generate the result from.
        min_length (int): The minimum length of the generated result.
        max_length (int): The maximum length of the generated result.

    Returns:
        str: The generated result.
    """
    result = MODEL_OBJECT.perform_inference(llm, text, min_length, max_length)
    return result


def make_messages(text: str) -> list:
    """Splits the given text into a list of messages by author."""
    escape_sequence = "###"
    text = re.sub(r"(Customer:\s|Agent:\s)", r"###\1", text)
    messages = text.split(escape_sequence)
    return messages


def make_sentences(text: str) -> list:
    """
    Splits the given text into a list of sentences.

    Args:
        text (str): The input text to split into sentences.

    Returns:
        list: A list of sentences extracted from the input text.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def make_llm() -> Pipeline:
    """
    Generates a language model (LLM) instance and saves it to a local directory.

    The function first checks if a pre-trained model exists in the "models/{model_name}" directory.
    If the pre-trained model exists, it uses that model. Otherwise, it generates a new model using the
    `MODEL_OBJECT.generate_model()` function.

    The generated model is then saved to the `models/{model_name}` directory for future use.

    Returns:
        LanguageModel: The generated language model instance.
    """
    model_name = MODEL_OBJECT.get_model_name()
    path = f"models/{model_name}"

    # Get pretrained from path if exists
    if os.path.exists(f"{path}"):
        MODEL_NAME_OR_PATH = path
    else:
        MODEL_NAME_OR_PATH = model_name

    llm = MODEL_OBJECT.generate_model(MODEL_NAME_OR_PATH, device=DEVICE)

    # Save LLM for future use
    if not os.path.exists(f"{path}"):
        Path(path).mkdir(parents=True, exist_ok=True)
        llm.save_pretrained(path)
    return llm


def format_prompt(text: str, system_prompt: str = "") -> str:
    """
    Formats a prompt by optionally prepending a system prompt to the given text.

    Args:
        text (str): The input text to format.
        system_prompt (str, optional): The system prompt to prepend to the text. Defaults to an empty string.

    Returns:
        str: The formatted prompt, with the system prompt prepended if provided.
    """
    if system_prompt.strip():
        return f"{system_prompt} {text}"
    return f"{text}"


def get_tokenizer(model_name: str) -> BartTokenizerFast:
    """Get tokenizer for model

    Args:
        model_name (str): Name of model to get tokenizer for.

    Returns:
        BartTokenizerFast: Tokenizer for model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def calculate_execution_time(start_time: float) -> str:
    """Calculate execution time

    Args:
        start_time (float): Start time of execution.

    Returns:
        str: Execution time with units appended.
    """
    time_taken = round(time.time() - start_time, 2)
    return f"{time_taken} seconds"
