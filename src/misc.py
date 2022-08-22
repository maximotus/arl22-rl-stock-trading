import logging
import os
import sys

import yaml as yaml


def parse_config(argv):
    try:
        config_path = argv[1]
        with open(config_path, "r") as stream:
            conf = yaml.safe_load(stream)
    except IndexError:
        print('Missing command line argument for the path to the configuration file. '
              'Please use this program like this: main.py PATH_TO_CONFIG_YAML_FILE')
        sys.exit()

    return config_path, conf


def setup_logger(path, lvl=0, fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"):
    """
    Sets up a global logger accessible via logging.getLogger('root').

    The registered logger will stream its outputs to the console as well as
    to a file out.log in the specified directory.

    Parameters
    ----------
    path : string
        Path to the folder in which to save the logfile out.log.
    lvl : int
        One of CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0.
    fmt : string
        Format string representing the format of the logs.
    """
    log_path = os.path.join(path, "out.log")

    fmt = "%(asctime)s - %(levelname)s - %(module)s - %(message)s" if fmt is None else fmt
    formatter = logging.Formatter(fmt=fmt)

    root_logger = logging.getLogger()

    lvl = 0 if lvl is None else lvl
    root_logger.setLevel(lvl)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger
