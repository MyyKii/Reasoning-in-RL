import logging

def setup_logging(
    log_file: str = "training.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
