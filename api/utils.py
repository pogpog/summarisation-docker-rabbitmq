import logging
import os

from dotenv import load_dotenv


def log(msg, level: int | str = "error"):
    """Logs a message with the specified log level.

    Args:
        msg (str): The message to log.
        level (str, int, optional): The log level. Can be one of `debug`, `info`,
            `warning`, or `error`. Alternatively, the int value can be provided directly.
            Defaults to `error`.
    """
    if type(level) == str:
        match level.lower():
            case "debug":
                level = logging.DEBUG
            case "info":
                level = logging.INFO
            case "warning":
                level = logging.WARNING
            case "error":
                level = logging.ERROR
            case _:
                level = logging.ERROR

    logging.basicConfig(
        filename="conversational-highlights.log",
        level=level,
        format="%(asctime)s : %(levelname)s : %(message)s",
    )

    logging.exception(msg)


def print_debug(msg):
    """Prints a message to the console if the `DEBUG` environment variable is set to `True`."""

    load_dotenv()
    if os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]:
        print(msg)
