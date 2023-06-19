from argparse import Namespace
import logging
from typing import Callable, Dict

from datasets import ClassLabel, DatasetDict
from datasets.arrow_dataset import Dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    Trainer,
)


CLASS_LABELS = ClassLabel(
    num_classes=5,
    names=[
        "Balance Sheets",
        "Cash Flow",
        "Income Statement",
        "Notes",
        "Others",
    ],
)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred: Callable[[EvalPrediction], Dict]) -> Dict:
    """
    Evaluation function to run at the end of every epoch
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(args: Namespace) -> None:
    dataset_dict = load_data(args)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path, num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.model_output_directory,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_training_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def load_data(config: Namespace) -> DatasetDict:
    train_data = Dataset.load_from_disk(config.sm_channel_train)
    test_data = Dataset.load_from_disk(config.sm_channel_test)
    # Remove the original text columns
    train_data = train_data.remove_columns("text")
    test_data = test_data.remove_columns("text")
    train_data = train_data.cast_column("label", CLASS_LABELS)
    test_data = test_data.cast_column("label", CLASS_LABELS)
    return DatasetDict({"train": train_data, "test": test_data})
