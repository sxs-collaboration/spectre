# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
import os

import click
import matplotlib.animation
import matplotlib.pyplot as plt
import rich.progress

logger = logging.getLogger(__name__)

DEFAULT_MPL_STYLESHEET = os.path.join(
    os.path.dirname(__file__), "plots.mplstyle"
)


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
            stylesheets = [DEFAULT_MPL_STYLESHEET]
            if stylesheet is not None:
                stylesheets.append(stylesheet)
            plt.style.use(stylesheets)
            return f(**kwargs)

        return command

    return decorator


def show_or_save_plot_command():
    """Add an 'output' CLI option and show or save the plot accordingly

    Apply this decorator to a CLI command that generates a plot or animation.
    Return the `matplotlib.Figure` or `matplotlib.animation.Animation` from your
    command. At the end of the command, the plot or animation is either shown
    interactively or saved to a file, depending on the user-specified 'output'
    option.
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
                "determines the file format, e.g. 'plot.png' or 'plot.pdf' for "
                "static plots and 'animation.gif' or 'animation.mp4' (requires "
                "ffmpeg) for animations. "
                "If no extension is given, the file format depends on the "
                "system settings (see matplotlib.pyplot.savefig docs)."
            ),
        )
        # Preserve the original function's name and docstring
        @functools.wraps(f)
        def command(output, **kwargs):
            # Call the original function
            fig_or_anim = f(**kwargs)
            # Show or save the plot
            if output:
                if isinstance(fig_or_anim, matplotlib.animation.Animation):
                    progress = rich.progress.Progress(
                        rich.progress.TextColumn(
                            "[progress.description]{task.description}"
                        ),
                        rich.progress.BarColumn(),
                        rich.progress.MofNCompleteColumn(),
                        rich.progress.TimeRemainingColumn(),
                    )
                    task_id = progress.add_task("Rendering frames", total=None)
                    with progress:
                        fig_or_anim.save(
                            output,
                            progress_callback=lambda i, n: progress.update(
                                task_id, completed=i + 1, total=n
                            ),
                        )
                else:
                    plt.savefig(output)
                    plt.close()
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
