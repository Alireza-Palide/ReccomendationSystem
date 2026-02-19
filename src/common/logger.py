import logging
import logging.config
import yaml
import os

def setup_logging(config_path="config/logging.yaml"):
    os.makedirs("logs", exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config file not found at {config_path}. Using basic config.")

def get_logger(name):
    return logging.getLogger(name)
