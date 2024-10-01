# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import rich.logging


def configure_logging(log_level: int):
    """
    Configure logging for our python scripts using the 'logging' module

    This is factored out into a free function so that any time we need to add
    module-specific logging configuration, we only have to add it to one place.

    Logging verbosity of the spectre module is set to the 'log_level'.
    For other modules we set the log level to INFO or above, so we don't get
    debug output from all the modules we import.
    """
    logging.basicConfig(
        level=max(log_level, logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler()],
    )
    logging.getLogger("spectre").setLevel(log_level)
