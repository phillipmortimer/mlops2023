from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    input_dir: str
    output_dir: str
    tokenizer: str
    test_split: float
