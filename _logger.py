import logging

loggers = {}


def setup_console_and_file_logger(name, log_file_name, level):
    """
    Create a logger with two handlers: (i) a log file handler and (ii) a console log handler.

    :param name: a name of the logger
    :param log_file_name: a file in which to store the log
    :param level: a minimum threshold severity level of messages that will be handled by this logger. This must be given
        in a format recognised by the logging.Handler.setLevel() function.
    :return a logger object with two handlers (file and console)
    """
    global loggers

    if loggers.get(name):
        return loggers.get(name)

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file_name, mode="w")
    file_handler.setFormatter(simple_formatter)
    file_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers = []

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    loggers.update(dict(name=logger))

    return logger
