import yaml

RANDOM_STATE = 42

with open('./config/fram.yaml', 'r') as f:
    CONFIG_F = yaml.safe_load(f)

with open('./config/uci.yaml', 'r') as f:
    CONFIG_U = yaml.safe_load(f)