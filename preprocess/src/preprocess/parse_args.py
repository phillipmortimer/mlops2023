import sys

from argparse import ArgumentParser

from preprocess.config import PreprocessConfig


def args_to_config() -> PreprocessConfig:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--input_dir",
        type=str,
        default="/opt/ml/processing/input/",
        help="Input directory for the data in the processing container",
    )

    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/ml/processing/output/",
        help="Output directory for the data in the processing container",
    )

    arg_parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of dataset to use as test set",
    )
    args = arg_parser.parse_args(sys.argv[1:])

    preprocess_config = PreprocessConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_split=args.test_split,
    )
    return preprocess_config
