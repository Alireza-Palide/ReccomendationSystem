import logging
import logging.config
import yaml
import os

def setup_logging(config_path="config/logging.yaml"):
    """
    Loads the logging configuration from the YAML file.
    Ensures the 'logs' directory exists.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback if config file is missing
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config file not found at {config_path}. Using basic config.")

def get_logger(name):
    """
    Returns a logger instance with the specified name.
    """
    return logging.getLogger(name)
