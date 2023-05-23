import os

from datasets import Dataset
from transformers import AutoTokenizer

from preprocess.config import PreprocessConfig
from preprocess.html_to_text import html_to_text


def build_dataset(config: PreprocessConfig) -> Dataset:
    # Directory names form the path labels
    # This code was written by GPT-4
    data = []
    for root, dirs, files in os.walk(config.input_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    label = os.path.basename(root)  # Use subdirectory name as the label
                    data.append({"text": text, "label": label})

    return Dataset.from_list(data)


def dataset_to_text(html_data: Dataset) -> Dataset:
    """
    Convert the html data to text
    """
    text_data = html_data.map(
        lambda x: {"text": html_to_text(x)}, input_columns="text", num_proc=8
    )
    return text_data


def tokenize(text_data: Dataset) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return text_data.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
