from typing import Dict

from sagemaker.experiments import Run
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class SagemakerExperimentsCallback(TrainerCallback):
    def __init__(self, run: Run) -> None:
        super().__init__()
        self.run = run

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs,
    ):
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.run.log_metric(name=name, value=value, step=int(state.epoch or 0))
        return super().on_log(args, state, control, logs=logs, **kwargs)
