#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np

from spectre.IO.Exporter import interpolate_to_points
from spectre.Visualization.OpenVolfiles import (
    open_volfiles_command,
    parse_point,
)
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)

logger = logging.getLogger(__name__)


def points_on_line(
    line_start: Sequence[float], line_end: Sequence[float], num_points: int
) -> np.ndarray:
    """Returns points on a line.

    Parameters:
      line_start: Start point of the line.
      line_end: End point of the line.
      num_points: Number of points to return.

    Returns: An array of shape (dim, num_points) with uniformly spaced points
      on the line.
    """
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_parameter = np.linspace(0, 1, num_points)
    normal = line_end - line_start
    return line_start[:, np.newaxis] + np.outer(normal, line_parameter)


@click.command(name="along-line")
@open_volfiles_command(obs_id_required=True, multiple_vars=True)
@click.option(
    "--line-start",
    "-A",
    callback=parse_point,
    help=(
        "Coordinates of the start of the line through the volume data. "
        "Specify as comma-separated list, e.g. '0,0,0'."
    ),
)
@click.option(
    "--line-end",
    "-B",
    callback=parse_point,
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
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_along_line_command(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    vars,
    line_start,
    line_end,
    num_samples,
    num_threads,
):
    """Plot variables along a line through volume data

    Interpolates the volume data in the H5_FILES to a line and plots the
    selected variables. You choose the line by specifying the start and end
    points.
    """
    # Interpolate the selected quantities to the line
    if line_start is None or line_end is None:
        raise click.UsageError(
            f"Specify '--line-start' / '-A' and '--line-end' / '-B'."
        )
    if len(line_start) != len(line_end):
        raise click.UsageError(
            "'--line-start' / '-A' and '--line-end' / '-B' must have"
            " the same dimension."
        )
    target_coords = points_on_line(line_start, line_end, num_samples)
    vars_on_line = interpolate_to_points(
        h5_files,
        subfile_name=subfile_name,
        observation_id=obs_id,
        tensor_components=vars,
        target_points=target_coords,
        num_threads=num_threads,
    )

    # Set x-axis for plot. For plotting along an axis, use the axis coordinate.
    # Otherwise, use the line parameter.
    normal = np.asarray(line_end) - np.asarray(line_start)
    nonzero_entries = np.nonzero(normal)[0]
    coord_axis = nonzero_entries[0] if len(nonzero_entries) == 1 else None
    if coord_axis is not None:
        x = target_coords[coord_axis]
        x_label = "xyz"[coord_axis]
    else:
        x = np.linspace(0, 1, num_samples)
        A_label = ", ".join(f"{x:g}" for x in line_start)
        B_label = ", ".join(f"{x:g}" for x in line_end)
        x_label = f"$({A_label})$ to $({B_label})$"

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


if __name__ == "__main__":
    plot_along_line_command(help_option_names=["-h", "--help"])
