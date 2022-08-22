import sys

from misc import setup_logger, parse_config


def main():
    configuration_file, configuration = parse_config(sys.argv)
    logger = setup_logger("./", 10)
    logger.info("Successfully read the given configuration file, created experiment directory and set up logger.")


if __name__ == "__main__":
    main()
