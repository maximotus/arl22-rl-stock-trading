import sys

from misc import setup_logger, parse_config, create_experiment_dir


def main():
    configuration_file, configuration = parse_config(sys.argv)

    mode = configuration.get("mode")
    log_lvl = configuration.get("logger").get("level")
    log_fmt = configuration.get("logger").get("format")
    experiment_path = create_experiment_dir(configuration_file, configuration.get("experiment_path"),
                                            configuration.get("model").get("pretrained_path"), mode)

    logger = setup_logger(experiment_path, log_lvl, log_fmt)
    logger.info("Successfully read the given configuration file, created experiment directory and set up logger.")
    logger.info('Starting experiment in mode ' + mode + ' using configuration ' + configuration_file)

    # TODO depending on mode, start training or evaluation
    raise NotImplementedError


if __name__ == "__main__":
    main()
