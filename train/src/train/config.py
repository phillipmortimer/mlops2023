from dataclasses import dataclass


@dataclass
class TrainConfig:
    sm_channel_training: str
    sm_channel_testing: str
    model_name_or_path: str
    model_output_directory: str
