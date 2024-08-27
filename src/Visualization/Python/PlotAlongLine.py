#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Sequence

import click
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

from spectre.IO.Exporter import interpolate_to_points
from spectre.Visualization.OpenVolfiles import (
    open_volfiles,
    open_volfiles_command,
    parse_point,
)
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from spectre.Visualization.ReadH5 import list_observations

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


def plot_along_line(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    vars,
    line_start,
    line_end,
    extrapolate_into_excisions=False,
    num_samples=200,
    num_threads=None,
    x_logscale=False,
    y_logscale=False,
    y_bounds=None,
    animate=False,
    interval=100,
):
    """Plot variables along a line through volume data

    Interpolates the volume data in the H5_FILES to a line and plots the
    selected variables. You choose the line by specifying the start and end
    points.

    Either select a specific observation in the volume data with '--step' or
    '--time', or specify '--animate' to produce an animation over all
    observations.
    """
    if animate == (obs_id is not None):
        raise click.UsageError(
            "Specify an observation '--step' or '--time', or specify"
            " '--animate' (but not both)."
        )
    if animate:
        obs_ids, obs_times = list_observations(
            open_volfiles(h5_files, subfile_name)
        )

    # Determine the line coordinates
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

    # Set up the figure
    fig, ax = plt.figure(), plt.gca()
    plt.xlim(x[0], x[-1])
    if x_logscale:
        plt.xscale("log")
    if y_logscale:
        plt.yscale("log")
    if y_bounds:
        plt.ylim(*y_bounds)
    plt.xlabel(x_label)
    # Keep track of plot lines to update in animation
    lines = [
        plt.plot([], [], label=var_name, **plot_kwargs)[0] for var_name in vars
    ]
    plt.legend()
    time_label = plt.annotate(
        "",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(4, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    def update_plot(obs_id, obs_time):
        vars_on_line = interpolate_to_points(
            h5_files,
            subfile_name=subfile_name,
            observation_id=obs_id,
            tensor_components=vars,
            target_points=target_coords,
            extrapolate_into_excisions=extrapolate_into_excisions,
            num_threads=num_threads,
        )
        for y, var_name, line in zip(vars_on_line, vars, lines):
            line.set_data(x, y)
        time_label.set_text(f"t = {obs_time:g}")
        ax.relim()
        ax.autoscale_view()

    # Animate or plot
    if animate:
        return matplotlib.animation.FuncAnimation(
            fig,
            lambda i: update_plot(obs_ids[i], obs_times[i]),
            init_func=list,
            frames=len(obs_ids),
            interval=interval,
            blit=False,
        )
    else:
        update_plot(obs_id, obs_time)
        return fig


@click.command(name="along-line", help=plot_along_line.__doc__)
@open_volfiles_command(obs_id_required=False, multiple_vars=True)
# Line options
# These aren't marked "required" so the user can omit them when using options
# like '--list-vars'.
@click.option(
    "--line-start",
    "-A",
    callback=parse_point,
    help=(
        "Coordinates of the start of the line through the volume data. "
        "Specify as comma-separated list, e.g. '0,0,0'.  [required]"
    ),
)
@click.option(
    "--line-end",
    "-B",
    callback=parse_point,
    help=(
        "Coordinates of the end of the line through the volume data. "
        "Specify as comma-separated list, e.g. '1,0,0'.  [required]"
    ),
)
@click.option(
    "--extrapolate-into-excisions",
    is_flag=True,
    help=(
        "Enables extrapolation into excision regions of the domain. "
        "This can be useful to fill the excision region with "
        "(constraint-violating but smooth) data so it can be imported into "
        "moving puncture codes."
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
@click.option("--x-logscale", is_flag=True, help="Set the x-axis to log scale.")
@click.option("--y-logscale", is_flag=True, help="Set the y-axis to log scale.")
@click.option(
    "--y-bounds",
    type=float,
    nargs=2,
    help="The lower and upper bounds of the y-axis.",
)
# Animation options
@click.option("--animate", is_flag=True, help="Animate over all observations.")
@click.option(
    "--interval",
    default=100,
    type=float,
    help="Delay between frames in milliseconds. Only used for animations.",
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_along_line_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    return plot_along_line(**kwargs)


if __name__ == "__main__":
    plot_along_line_command(help_option_names=["-h", "--help"])
