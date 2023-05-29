import logging

from datasets import ClassLabel, DatasetDict
from datasets.arrow_dataset import Dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from train.config import TrainConfig

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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(config: TrainConfig):
    dataset_dict = load_data(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=config.model_output_directory,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
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


def load_data(config) -> DatasetDict:
    train_data = Dataset.load_from_disk(config.sm_channel_training)
    test_data = Dataset.load_from_disk(config.sm_channel_testing)
    # Remove the original text columns
    train_data = train_data.remove_columns("text")
    test_data = test_data.remove_columns("text")
    train_data = train_data.cast_column("label", CLASS_LABELS)
    test_data = test_data.cast_column("label", CLASS_LABELS)
    return DatasetDict({"train": train_data, "test": test_data})
