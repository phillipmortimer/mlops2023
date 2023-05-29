import sys

from argparse import ArgumentParser

from train.config import Config


def args_to_config() -> Config:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--sm_channel_training",
        type=str,
        default="/opt/ml/input/data/training",
        help="Input directory for the training data in the container",
    )

    arg_parser.add_argument(
        "--sm_channel_testing",
        type=str,
        default="/opt/ml/input/data/testing",
        help="Output directory for the testing data in the container",
    )
    arg_parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/opt/ml/input/data/weights",
        help="Model name in the Huggingface hub, or path to local weights",
    )
    arg_parser.add_argument(
        "--model_output_directory",
        type=str,
        default="/opt/ml/model",
        help="Model name in the Huggingface hub, or path to local weights",
    )

    args = arg_parser.parse_args(sys.argv[1:])

    train_config = Config(
        sm_channel_training=args.sm_channel_training,
        sm_channel_testing=args.sm_channel_testing,
    )
    return train_config
