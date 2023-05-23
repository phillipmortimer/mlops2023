import logging
import os

from transformers import AutoTokenizer

from preprocess.build_dataset import build_dataset
from preprocess.html_to_text import html_to_text
from preprocess.parse_args import args_to_config


logger = logging.getLogger(__name__)


def preprocess():
    config = args_to_config()

    logger.info("Reading dataset from disk")
    dataset = build_dataset(config)

    text_data = dataset.map(
        lambda x: {"text": html_to_text(x)}, input_columns="text", num_proc=4
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_dataset = text_data.map(
        lambda x: tokenizer(x["text"], truncation=True), batched=True
    )

    dataset_dict = tokenized_dataset.train_test_split(
        test_size=config.test_split, shuffle=True
    )

    logger.info("Writing training dataset to disk")
    dataset_dict["train"].save_to_disk(os.path.join(config.output_dir, "train"))
    logger.info("Writing test dataset to disk")
    dataset_dict["test"].save_to_disk(os.path.join(config.output_dir, "test"))
