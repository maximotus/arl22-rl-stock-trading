import logging
import os
import shutil
import sys

import yaml as yaml

from datetime import datetime


def create_experiment_dir(conf_file, exp_path, pretrained_path, run_mode):
    """
    Creates a directory for all the outputs of the experiment (i.e. program execution).

    The directory is of shape exp_path/run_mode/config_name/timestamp_now/ (= base_path) with
    subdirectories /model, /results and /stats.
    Also makes a snapshot of the configuration file and saves it to the created experiment directory.

    Parameters
    ----------
    conf_file : string
        Path / name of the configuration file.
    exp_path : string
        Path of the experiment directory.
    pretrained_path : string
        Path of any pretrained model.
    run_mode : string
        Run mode (has to be in [train, eval]).

    Returns
    -------
    base_path : string
        The base path of the experiment directory.
    """
    valid_modes = ["train", "eval", "gen"]

    if pretrained_path is not None:
        base_path = pretrained_path.replace("train", run_mode, 1)
    else:
        config_name = conf_file[10:-5]
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base_path = os.path.join(exp_path, run_mode, config_name, now)

    paths = [os.path.join(base_path, "results")]
    if run_mode in valid_modes[0:2]:
        paths.append(os.path.join(base_path, "stats"))
    if run_mode == valid_modes[0]:
        paths.append(os.path.join(base_path, "model"))

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Created directory", path)

    shutil.copy2(conf_file, base_path)
    print("Copied config file", conf_file, "to", base_path)

    return base_path


def parse_config(argv):
    """
    Parses the configuration file located at the path that is given with the first command line argument.

    If the required command line argument is missing, the program stops.

    Parameters
    ----------
    argv : list[string]
        Command line arguments.

    Returns
    -------
    config_path : string
        The path to the configuration file.
    conf : dict
        The dictionary representing the configuration.
    """
    try:
        config_path = argv[1]
        with open(config_path, "r") as stream:
            conf = yaml.safe_load(stream)
    except IndexError:
        print("Missing command line argument for the path to the configuration file. "
              "Please use this program like this: main.py PATH_TO_CONFIG_YAML_FILE")
        sys.exit()

    return config_path, conf


def setup_logger(path, lvl=0, fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"):
    """
    Sets up a global logger accessible via logging.getLogger("root").

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

    Returns
    -------
    root_logger : Logger
        The root logger.
    """
    log_path = os.path.join(path, "out.log")
    formatter = logging.Formatter(fmt=fmt)
    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger
