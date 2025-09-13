import yaml


def load_config(config_path="config.yaml"):
    """
    Loads the configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
