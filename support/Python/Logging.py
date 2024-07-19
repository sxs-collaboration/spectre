# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import rich.logging


def configure_logging(log_level=logging.INFO):
    """
    Configure logging for our python scripts using the 'logging' module

    This is factored out into a free function so that any time we need to add
    module-specific logging configuration, we only have to add it to one place.

    Module specific logging info:
     - Disable 'matplotlib.font_manager' logging for 'logging.DEBUG' or higher
    """
    logging.basicConfig(
        level=logging.INFO if log_level is None else log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler()],
    )
    if log_level is not None and log_level >= logging.DEBUG:
        logging.getLogger("matplotlib.font_manager").disabled = True
