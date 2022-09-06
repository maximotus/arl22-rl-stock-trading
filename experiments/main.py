import argparse

from experiment import TrainExperiment
from misc import setup_logger, parse_config, create_experiment_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        metavar="PATH_TO_CONF_FILE",
        required=True,
        help="relative or absolute path to the configuration file",
    )
    args = parser.parse_args()

    # read configuration file and finnhub api key
    configuration_file = args.conf
    configuration = parse_config(configuration_file)

    # experiment setup
    mode = configuration.get("mode")
    log_lvl = configuration.get("logger").get("level")
    log_fmt = configuration.get("logger").get("format")
    experiment_path = create_experiment_dir(
        configuration_file,
        configuration.get("experiment_path"),
        configuration.get("agent").get("model").get("pretrained_path"),
        mode,
    )

    # overwrite overall experiment path with the newly created base_path of the experiment
    configuration["experiment_path"] = experiment_path

    # logger setup
    logger = setup_logger(experiment_path, log_lvl, log_fmt)
    logger.info(
        "Successfully read the given configuration file, created experiment directory and set up logger."
    )
    logger.info(
        f"Starting experiment in mode {mode} using configuration {configuration_file}"
    )

    if mode == "train":
        exp = TrainExperiment(configuration)
        exp.conduct()

    if mode == "eval":
        raise NotImplementedError


if __name__ == "__main__":
    main()
