#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.IO.Exporter import interpolate_to_points

logger = logging.getLogger(__name__)


def _parse_step(ctx, param, value):
    if value is None:
        return None
    if value.lower() == "first":
        return 0
    if value.lower() == "last":
        return -1
    return int(value)


def _parse_point(ctx, param, value):
    if not value:
        return None
    return np.array(list(map(float, value.split(","))))


@click.command(name="plot-along-line")
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help="Name of subfile within h5 file containing volume data to plot.",
)
@click.option(
    "--list-vars",
    "-l",
    is_flag=True,
    help="Print available variables and exit.",
)
@click.option(
    "--var",
    "-y",
    "vars",
    multiple=True,
    help=(
        "Variables to plot. List any tensor components "
        "in the volume data file, such as 'Shift_x'."
    ),
)
@click.option(
    "--step",
    callback=_parse_step,
    help="Observation step number. Specify '-1' for the last step in the file.",
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
    "--line-start",
    "-A",
    callback=_parse_point,
    help=(
        "Coordinates of the start of the line through the volume data. "
        "Specify as comma-separated list, e.g. '0,0,0'."
    ),
)
@click.option(
    "--line-end",
    "-B",
    callback=_parse_point,
    help=(
        "Coordinates of the end of the line through the volume data. "
        "Specify as comma-separated list, e.g. '1,0,0'."
    ),
)
@click.option(
    "--num-samples",
    "-N",
    type=int,
    default=200,
    show_default=True,
    help=(
        "Number of uniformly spaced samples along the line to which volume "
        "data is interpolated."
    ),
)
@click.option(
    "--num-threads",
    "-j",
    type=int,
    show_default="all available cores",
    help=(
        "Number of threads to use for interpolation. Only available if compiled"
        " with OpenMP. Parallelization is over volume data files, so this only"
        " has an effect if multiple files are specified."
    ),
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
def plot_along_line_command(
    h5_files,
    subfile_name,
    list_vars,
    vars,
    step,
    output,
    line_start,
    line_end,
    num_samples,
    num_threads,
    stylesheet,
):
    """Plot variables along a line through volume data

    Interpolates the volume data in the H5_FILES to a line and plots the
    selected variables. You choose the line by specifying the start and end
    points.
    """
    # Script should be a noop if input files are empty
    if not h5_files:
        return

    # Open first H5 file to get some info
    open_h5_file = spectre_h5.H5File(h5_files[0], "r")

    # Print available subfile names and exit
    if not subfile_name:
        import rich.columns

        available_subfiles = open_h5_file.all_vol_files()
        if len(available_subfiles) == 1:
            subfile_name = available_subfiles[0]
        else:
            rich.print(rich.columns.Columns(available_subfiles))
            return

    # Normalize subfile name
    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]
    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name

    volfile = open_h5_file.get_vol(subfile_name)
    obs_ids = volfile.list_observation_ids()
    obs_values = list(map(volfile.get_observation_value, obs_ids))
    dim = volfile.get_dimension()

    # Select observation
    if step is None:
        raise click.UsageError(
            f"Must specify '--step' (in [0, {len(obs_ids) - 1}], or -1)."
        )
    obs_id = obs_ids[step]

    # Print available variables and exit
    all_vars = volfile.list_tensor_components(obs_id)
    if list_vars or not vars:
        import rich.columns

        rich.print(rich.columns.Columns(all_vars))
        return
    for var in vars:
        if var not in all_vars:
            raise click.UsageError(
                f"Unknown variable '{var}'. Available variables are: {all_vars}"
            )

    # Close the H5 file because we're done with preprocessing
    open_h5_file.close()

    # Interpolate the selected quantities to the line
    if line_start is None or line_end is None:
        raise click.UsageError(
            f"Specify '--line-start' / '-A' and '--line-end' / '-B'."
        )
    if len(line_start) != dim or len(line_end) != dim:
        raise click.UsageError(
            "Both '--line-start' / '-A' and '--line-end' / '-B' must have"
            f" {dim} values (the dimension of the volume data)."
        )
    line_parameter = np.linspace(0, 1, num_samples)
    normal = line_end - line_start
    target_coords = line_start + np.outer(line_parameter, normal)
    vars_on_line = interpolate_to_points(
        h5_files,
        subfile_name=subfile_name,
        observation_step=step,
        tensor_components=vars,
        target_points=target_coords.T,
        num_threads=num_threads,
    )

    # Set x-axis for plot. For plotting along an axis, use the axis coordinate.
    # Otherwise, use the line parameter.
    if (dim < 2 or normal[1] == 0) and (dim < 3 or normal[2] == 0):
        x = target_coords[:, 0]
        x_label = "x"
    elif normal[0] == 0 and (dim < 3 or normal[2] == 0):
        x = target_coords[:, 1]
        x_label = "y"
    elif dim == 3 and normal[0] == 0 and normal[1] == 0:
        x = target_coords[:, 2]
        x_label = "z"
    else:
        x = line_parameter
        A_label = ", ".join(f"{x:g}" for x in line_start)
        B_label = ", ".join(f"{x:g}" for x in line_end)
        x_label = f"$({A_label})$ to $({B_label})$"

    # Apply stylesheet
    if stylesheet is not None:
        plt.style.use(stylesheet)

    # Select plotting parameters. Any further customization of the plotting
    # style can be done with a stylesheet.
    plot_kwargs = dict(
        color="black" if len(vars) == 1 else None,
    )

    # Plot the selected quantities
    for y, var_name in zip(vars_on_line, vars):
        plt.plot(
            x,
            y,
            label=var_name,
            **plot_kwargs,
        )

    # Configure the axes
    plt.xlim(x[0], x[-1])
    plt.xlabel(x_label)
    plt.legend()

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
    plot_along_line_command(help_option_names=["-h", "--help"])
