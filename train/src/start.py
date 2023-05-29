from train.parse_args import args_to_config
from train.train import train


config = args_to_config()
train(config)
