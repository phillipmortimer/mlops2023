import boto3

from sagemaker.experiments import load_run
from sagemaker.session import Session

from train.parse_args import args_to_config
from train.train import train


config = args_to_config()

sagemaker_session = Session(
    boto3.session.Session(
        region_name=config.region,
    )
)
run = load_run(sagemaker_session=sagemaker_session)

with run:
    run.log_parameters(vars(config))
    train(config, run)
