# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

# this script lives in ch05/, so the repository root has to be importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tiktoken
import torch
import chainlit
from utils.paths import repo_path
from utils.previous_chapters import GPTModel
from utils.previous_chapters import classify_review

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_tokenizer():
    """
    Code to load the finetuned GPT-2 model generated in chapter 5.
    This requires that you run the code in ch05/ch05.ipynb first, which generates
    the necessary review_classifier.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = repo_path("ch05", "review_classifier.pth")
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 5 code"
            " (ch05.ipynb) to generate the review_classifier.pth file."
        )
        sys.exit()

    # Instantiate model
    model = GPTModel(GPT_CONFIG_124M)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # Then load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model

# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model = get_model_and_tokenizer()

@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # This returns the model response to the interface
    ).send()