#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Sequence, Union

import click
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
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


def points_on_slice(
    slice_origin: Sequence[float],
    slice_extent: Sequence[float],
    slice_normal: Sequence[float],
    slice_up: Sequence[float],
    num_points: Sequence[int],
):
    """Returns points on a rectangular slice in 3D.

    Parameters:
      slice_origin: Coordinates of the center of the slice.
      slice_extent: Coordinate extent in both directions of the slice (two
        numbers).
      slice_normal: Direction of the normal of the slice.
      slice_up: Up-direction of the slice.
      num_points: Number of points to return in each dimension (two numbers).

    Returns: An array of shape (3, num_points[0], num_points[1]) with uniformly
      spaced points on the slice.
    """
    assert np.isclose(
        np.dot(slice_up, slice_normal), 0.0
    ), "Up and normal vectors must be orthogonal"
    assert (
        len(num_points) == 2
    ), "Specify two numbers for num_points (one per dimension of the slice)"
    assert (
        len(slice_extent) == 2
    ), "Specify two numbers for slice_extent (one per dimension of the slice)"
    n = slice_normal / np.linalg.norm(slice_normal)
    u = slice_up / np.linalg.norm(slice_up)
    v = np.cross(u, n) * slice_extent[0]
    u *= slice_extent[1]
    lower_left = slice_origin - 0.5 * (u + v)
    xx, yy = np.meshgrid(
        np.linspace(0, 1, num_points[0]),
        np.linspace(0, 1, num_points[1]),
        indexing="ij",
    )
    return (
        lower_left[:, np.newaxis, np.newaxis]
        + v[:, np.newaxis, np.newaxis] * xx[np.newaxis, :, :]
        + u[:, np.newaxis, np.newaxis] * yy[np.newaxis, :, :]
    )


def plot_slice(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    var_name,
    slice_origin,
    slice_extent,
    slice_normal,
    slice_up,
    extrapolate_into_excisions=False,
    num_samples=[200, 200],
    num_threads=None,
    title=None,
    data_bounds=None,
    animate=False,
    interval=100,
):
    """Plot variables on a slice through volume data

    Interpolates the volume data in the H5_FILES to a slice and plots the
    selected variables. You choose the slice by specifying its center, extents,
    normal, and up direction.

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

    # Determine the slice coordinates
    if (
        slice_origin is None
        or slice_extent is None
        or slice_normal is None
        or slice_up is None
    ):
        raise click.UsageError(
            f"Specify all slice parameters: '--slice-[origin,extent,normal,up]'"
        )
    target_coords = points_on_slice(
        slice_origin, slice_extent, slice_normal, slice_up, num_samples
    )
    dim = target_coords.shape[0]
    assert dim == 3, "Only 3D slices are supported"

    # Set axes for plot. For plotting along an axis, use the axis coordinate.
    # Otherwise, use the affine parameter.
    def coord_axis(vec):
        nonzero_entries = np.nonzero(vec)[0]
        if len(nonzero_entries) == 1:
            return nonzero_entries[0]
        return None

    y_axis = coord_axis(slice_up)
    x_axis = coord_axis(np.cross(slice_up, slice_normal))
    if x_axis is not None:
        x = target_coords[x_axis]
        x_label = "xyz"[x_axis]
    else:
        x = np.linspace(0, 1, num_samples[0])
        x_label = None
    if y_axis is not None:
        y = target_coords[y_axis]
        y_label = "xyz"[y_axis]
    else:
        y = np.linspace(0, 1, num_samples[1])
        y_label = None

    # Determine global data bounds. Only needed for an animation.
    if data_bounds is None and animate:
        data_bounds = [np.inf, -np.inf]
        for volfile in open_volfiles(h5_files, subfile_name):
            for obs_id in volfile.list_observation_ids():
                data = volfile.get_tensor_component(obs_id, var_name).data
                data_bounds[0] = min(data_bounds[0], data.min())
                data_bounds[1] = max(data_bounds[1], data.max())
        logger.info(
            f"Determined data bounds for '{var_name}':"
            f" [{data_bounds[0]:g}, {data_bounds[1]:g}]"
        )
    norm = (
        plt.Normalize(vmin=data_bounds[0], vmax=data_bounds[1])
        if data_bounds
        else None
    )
    levels = (
        np.linspace(data_bounds[0], data_bounds[1], 10) if data_bounds else 10
    )

    # Set up the figure
    fig, ax = plt.figure(), plt.gca()
    plt.title(title or var_name)
    if np.isclose(slice_extent[0], slice_extent[1]):
        plt.gca().set_aspect("equal")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm), ax=ax)
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

    # Interpolate data and plot the slice
    def plot_slice(obs_id, time):
        data = np.array(
            interpolate_to_points(
                h5_files,
                subfile_name=subfile_name,
                observation_id=obs_id,
                tensor_components=[var_name],
                target_points=target_coords.reshape(3, np.prod(num_samples)),
                extrapolate_into_excisions=extrapolate_into_excisions,
                num_threads=num_threads,
            )[0]
        ).reshape(num_samples)
        contours_filled = plt.contourf(
            x, y, data, levels=levels, norm=norm, extend="both"
        )
        contours = plt.contour(
            contours_filled, colors="white", linewidths=0.5, alpha=0.6
        )
        time_label.set_text(f"t = {time:g}")
        return contours_filled.collections + contours.collections

    # Plot a static slice and return early
    if not animate:
        plot_slice(obs_id, obs_time)
        return fig

    # Animate the slice
    # Keep track of old artists to clear before plotting new ones
    artists = []

    def update_plot(frame):
        nonlocal artists
        # Clear old artists
        for artist in artists:
            artist.remove()
        # Plot new slice
        artists = plot_slice(obs_ids[frame], obs_times[frame])

    return matplotlib.animation.FuncAnimation(
        fig,
        update_plot,
        init_func=list,
        frames=len(obs_ids),
        interval=interval,
        blit=False,
    )


@click.command(name="slice", help=plot_slice.__doc__)
@open_volfiles_command(obs_id_required=False, multiple_vars=False)
# Slice options
# These aren't marked "required" so the user can omit them when using options
# like '--list-vars'.
@click.option(
    "--slice-origin",
    "--slice-center",
    "-C",
    callback=parse_point,
    help=(
        "Coordinates of the center of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,0,0'.  [required]"
    ),
)
@click.option(
    "--slice-extent",
    "-X",
    type=float,
    nargs=2,
    help=(
        "Extent in both directions of the slice through the volume data, e.g. "
        "'-X 10 10' for a 10x10 slice in the coordinates of the volume data."
        "  [required]"
    ),
)
@click.option(
    "--slice-normal",
    "-n",
    callback=parse_point,
    help=(
        "Direction of the normal of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,0,1' for a slice "
        "in the xy-plane.  [required]"
    ),
)
@click.option(
    "--slice-up",
    "-u",
    callback=parse_point,
    help=(
        "Up-direction of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,1,0' so the y-axis "
        "is the vertical axis of the plot.  [required]"
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
    nargs=2,
    default=(200, 200),
    show_default=True,
    help=(
        "Number of uniformly spaced samples along each direction of the slice "
        "to which volume data is interpolated."
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
    "--title",
    "-t",
    help="Title for the plot.",
    show_default="name of the variable",
)
@click.option(
    "--y-bounds",
    "--data-bounds",
    "data_bounds",
    type=float,
    nargs=2,
    help="Lower and upper bounds for the color scale of the plot.",
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
def plot_slice_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    return plot_slice(**kwargs)


if __name__ == "__main__":
    plot_slice_command(help_option_names=["-h", "--help"])
