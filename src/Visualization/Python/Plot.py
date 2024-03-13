# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
import os

import click
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def apply_stylesheet_command():
    """Add a CLI option to apply a stylesheet for plotting"""

    def decorator(f):
        @click.option(
            "--stylesheet",
            "-s",
            envvar="SPECTRE_MPL_STYLESHEET",
            help=(
                "Select a matplotlib stylesheet for customization of the plot,"
                " such as linestyle cycles, linewidth, fontsize, legend, etc."
                " Specify a filename or one of the built-in styles. See"
                " https://matplotlib.org/gallery/style_sheets/style_sheets_reference"
                " for a list of built-in styles, e.g. 'seaborn-dark'. The"
                " stylesheet can also be set with the 'SPECTRE_MPL_STYLESHEET'"
                " environment variable."
            ),
        )
        # Preserve the original function's name and docstring
        @functools.wraps(f)
        def command(stylesheet, **kwargs):
            # Apply the default stylesheet and the user-provided one
            stylesheets = [
                os.path.join(os.path.dirname(__file__), "plots.mplstyle")
            ]
            if stylesheet is not None:
                stylesheets.append(stylesheet)
            plt.style.use(stylesheets)
            return f(**kwargs)

        return command

    return decorator


def show_or_save_plot_command():
    """Add an 'output' CLI option and show or save the plot accordingly

    Apply this decorator to a CLI command that generates a plot. At the end of
    the command, the plot is either shown interactively or saved to a file,
    depending on the user-specified 'output' option.
    """

    def decorator(f):
        @click.option(
            "--output",
            "-o",
            type=click.Path(file_okay=True, dir_okay=False, writable=True),
            help=(
                "Name of the output plot file. If unspecified, the plot is "
                "shown interactively, which only works on machines with a "
                "window server. If a filename is specified, its extension "
                "determines the file format, e.g. 'plot.png' or 'plot.pdf'. "
                "If no extension is given, the file format depends on the "
                "system settings (see matplotlib.pyplot.savefig docs)."
            ),
        )
        # Preserve the original function's name and docstring
        @functools.wraps(f)
        def command(output, **kwargs):
            # Call the original function
            f(**kwargs)
            # Show or save the plot
            if output:
                plt.savefig(output)
            else:
                if not os.environ.get("DISPLAY"):
                    logger.warning(
                        "No 'DISPLAY' environment variable is configured so"
                        " plotting interactively is unlikely to work. Write the"
                        " plot to a file with the --output/-o option."
                    )
                plt.show()

        return command

    return decorator
