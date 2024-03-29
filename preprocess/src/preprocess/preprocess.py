import logging
import os

from transformers import AutoTokenizer

from preprocess.build_dataset import build_dataset
from preprocess.html_to_text import html_to_text
from preprocess.parse_args import args_to_config


logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def preprocess():
    config = args_to_config()

    logger.info("Reading dataset from disk")
    dataset = build_dataset(config)

    text_data = dataset.map(
        lambda x: {"text": html_to_text(x)}, input_columns="text", num_proc=4
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    tokenized_dataset = text_data.map(
        lambda x: tokenizer(x["text"], truncation=True), batched=True
    )

    dataset_dict = tokenized_dataset.train_test_split(
        test_size=config.test_split, shuffle=True
    )

    train_output = os.path.join(config.output_dir, "train")
    logger.info(f"Writing training dataset to {train_output}")
    dataset_dict["train"].save_to_disk(train_output)
    test_output = os.path.join(config.output_dir, "test")
    logger.info(f"Writing test dataset to disk {test_output}")
    dataset_dict["test"].save_to_disk(test_output)
