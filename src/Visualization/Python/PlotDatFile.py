#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rich

from spectre.Visualization.ReadH5 import available_subfiles

logger = logging.getLogger(__name__)


def parse_functions(ctx, param, all_values):
    """Parse function names and their labels

    Functions and their labels can be specified as key-value pairs such as
    'Error(ScalarField)=$L_2(\\phi)$'. Remember to wrap the key-value pair in
    quotes on the command line to avoid issues with special characters or
    spaces.
    """
    if all_values is None:
        return {}
    functions_and_labels = {}
    for value in all_values:
        key_and_value = value.split("=")
        if len(key_and_value) == 1:
            # No label specified, use name of function as label
            functions_and_labels[key_and_value[0]] = key_and_value[0]
        elif len(key_and_value) == 2:
            functions_and_labels[key_and_value[0]] = key_and_value[1]
        else:
            raise click.BadParameter(
                f"The value of '{value}' could not be parsed as a key-value "
                "pair. It should have a single '=' or none."
            )
    return functions_and_labels


@click.command()
@click.argument(
    "h5_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help=(
        "The dat subfile to read. "
        "If unspecified, all available dat subfiles will be printed."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help=(
        "Name of the output plot file. If unspecified, the plot is "
        "shown interactively, which only works on machines with a "
        "window server."
    ),
)
@click.option(
    "--legend-only",
    "-l",
    is_flag=True,
    help="Print out the available quantities and exit.",
)
@click.option(
    "--functions",
    "-y",
    multiple=True,
    callback=parse_functions,
    help=(
        "The quantities to plot. If unspecified, list all available "
        "quantities and exit."
    ),
)
@click.option(
    "--x-axis",
    "-x",
    help="Select the column in the dat file uses as the x-axis in the plot.",
    show_default="first column in the dat file",
)
# Plotting options
@click.option(
    "--x-label",
    help="The label on the x-axis.",
    show_default="name of the x-axis column",
)
@click.option(
    "--y-label",
    required=False,
    help="The label on the y-axis.",
    show_default="no label",
)
@click.option("--x-logscale", is_flag=True, help="Set the x-axis to log scale.")
@click.option("--y-logscale", is_flag=True, help="Set the y-axis to log scale.")
@click.option(
    "--x-bounds",
    type=float,
    nargs=2,
    help="The lower and upper bounds of the x-axis.",
)
@click.option(
    "--y-bounds",
    type=float,
    nargs=2,
    help="The lower and upper bounds of the y-axis.",
)
@click.option(
    "--title", "-t", help="Title of the graph.", show_default="subfile name"
)
@click.option(
    "--stylesheet",
    "-s",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    envvar="SPECTRE_MPL_STYLESHEET",
    help=(
        "Select a matplotlib stylesheet for customization of the plot, such "
        "as linestyle cycles, linewidth, fontsize, legend, etc. "
        "The stylesheet can also be set with the 'SPECTRE_MPL_STYLESHEET' "
        "environment variable."
    ),
)
def plot_dat_command(
    h5_file,
    subfile_name,
    output,
    legend_only,
    functions,
    x_axis,
    stylesheet,
    x_label,
    y_label,
    x_logscale,
    y_logscale,
    x_bounds,
    y_bounds,
    title,
):
    """Plot columns in '.dat' datasets in H5 files"""
    _rich_traceback_guard = True  # Hide traceback until here

    with h5py.File(h5_file, "r") as h5file:
        # Print available subfiles and exit
        if subfile_name is None:
            import rich.columns

            rich.print(
                rich.columns.Columns(
                    available_subfiles(h5file, extension=".dat")
                )
            )
            return

        # Open subfile
        if not subfile_name.endswith(".dat"):
            subfile_name += ".dat"
        dat_file = h5file.get(subfile_name)
        if dat_file is None:
            raise click.UsageError(
                f"Unable to open dat file '{subfile_name}'. Available "
                f"files are:\n {available_subfiles(h5file, extension='.dat')}"
            )

        # Read legend from subfile
        legend = list(dat_file.attrs["Legend"])

        # Select x-axis
        if x_axis is None:
            x_axis = legend[0]
        elif x_axis not in legend:
            raise click.UsageError(
                f"Unknown x-axis '{x_axis}'. Available columns are: {legend}"
            )

        # Print legend and exit if requested
        if legend_only or not functions:
            import rich.table

            rich.print(f"{len(legend)} columns x {len(dat_file)} data points:")
            table = rich.table.Table(show_header=False, box=None)
            for i, function in enumerate(legend):
                table.add_row(
                    # Is the quantity selected as x or y?
                    (
                        "x"
                        if function == x_axis
                        else "y" if function in functions else ""
                    ),
                    # Name of the quantity
                    function,
                    # Value bounds
                    f"[{np.min(dat_file[:, i]):g}, {np.max(dat_file[:, i]):g}]",
                    # Highlight selected quantities
                    style=(
                        "bold"
                        if function == x_axis or function in functions
                        else None
                    ),
                )
            rich.print(table)
            return

        # Apply stylesheet
        if stylesheet is not None:
            plt.style.use(stylesheet)

        # Select plotting parameters. Any further customization of the plotting
        # style can be done with a stylesheet.
        plot_kwargs = dict(
            color="black" if len(functions) == 1 else None,
            marker="." if len(dat_file) < 20 else None,
        )

        # Plot the selected quantities
        for function, label in functions.items():
            if function not in legend:
                raise click.UsageError(
                    f"Unknown function '{function}'. "
                    f"Available functions are: {legend}"
                )

            plt.plot(
                dat_file[:, legend.index(x_axis)],
                dat_file[:, legend.index(function)],
                label=label,
                **plot_kwargs,
            )

    # Configure the axes
    if y_logscale:
        plt.yscale("log")
    if x_logscale:
        plt.xscale("log")
    plt.xlabel(x_label if x_label else x_axis)
    if y_label:
        plt.ylabel(y_label)
    plt.legend()
    if x_bounds:
        plt.xlim(*x_bounds)
    if y_bounds:
        plt.ylim(*y_bounds)
    plt.title(title if title else subfile_name[:-4])

    if output:
        plt.savefig(output, format="pdf", bbox_inches="tight")
    else:
        if not os.environ.get("DISPLAY"):
            logger.warning(
                "No 'DISPLAY' environment variable is configured so plotting "
                "interactively is unlikely to work. Write the plot to a file "
                "with the --output/-o option."
            )
        plt.show()


if __name__ == "__main__":
    plot_dat_command(help_option_names=["-h", "--help"])
