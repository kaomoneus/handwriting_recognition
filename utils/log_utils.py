import logging
from logging.config import dictConfig
from os.path import exists

import yaml


import logging
import tqdm


# TODO: use this handler instead of StreamHandler
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def init_logging():
    log_config_path = "log.yaml"
    log_config = None
    if exists(log_config_path):
        with open(log_config_path) as flog:
            log_config = yaml.safe_load(flog)
        dictConfig(log_config)

    if log_config:
        logging.getLogger(__name__).debug(f"Loaded LOGs config from '{log_config}'.")

    logging.getLogger(__name__).debug("LOGs initialized.")
