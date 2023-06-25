from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import os
import sys

from transformers import TrainingArguments


@dataclass
class TrainConfig:
    sm_channel_training: str
    sm_channel_testing: str
    training_arguments: TrainingArguments


def args_to_config() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--sm_channel_train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN") or "/opt/ml/input/data/training",
        help="Input directory for the training data in the container",
    )

    arg_parser.add_argument(
        "--sm_channel_test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST") or "/opt/ml/input/data/testing",
        help="Input directory for the testing data in the container",
    )
    arg_parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/opt/ml/input/data/weights",
        help="Model name in the Huggingface hub, or path to local weights",
    )
    arg_parser.add_argument(
        "--model_output_directory",
        type=str,
        default=os.environ.get("SM_MODEL_DIR") or "/opt/ml/model",
        help="Directory into which to write the model artifacts after training",
    )
    arg_parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )
    arg_parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
    )
    arg_parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
    )
    arg_parser.add_argument("--weight_decay", type=float, default=0.01)
    arg_parser.add_argument(
        "--region",
        type=str,
        default="eu-west-3",
    )

    return arg_parser.parse_args(sys.argv[1:])
