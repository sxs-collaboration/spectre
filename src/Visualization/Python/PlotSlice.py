#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Sequence, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
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


@click.command(name="slice")
@open_volfiles_command(obs_id_required=True, multiple_vars=False)
@click.option(
    "--slice-origin",
    "--slice-center",
    "-C",
    callback=parse_point,
    help=(
        "Coordinates of the center of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,0,0'."
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
    ),
)
@click.option(
    "--slice-normal",
    "-n",
    callback=parse_point,
    help=(
        "Direction of the normal of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,0,1' for a slice "
        "in the xy-plane."
    ),
)
@click.option(
    "--slice-up",
    "-u",
    callback=parse_point,
    help=(
        "Up-direction of the slice through the volume "
        "data. Specify as comma-separated list, e.g. '0,1,0' so the y-axis "
        "is the vertical axis of the plot."
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
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_slice_command(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    var_name,
    slice_origin,
    slice_extent,
    slice_normal,
    slice_up,
    num_samples,
    num_threads,
    title,
):
    """Plot variables on a slice through volume data

    Interpolates the volume data in the H5_FILES to a slice and plots the
    selected variables. You choose the slice by specifying three corners of a
    rectangle.
    """
    # Interpolate the selected variable to the slice
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
    data = np.array(
        interpolate_to_points(
            h5_files,
            subfile_name=subfile_name,
            observation_id=obs_id,
            tensor_components=[var_name],
            target_points=target_coords.reshape(3, np.prod(num_samples)),
            num_threads=num_threads,
        )[0]
    ).reshape(num_samples)

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

    # Plot the slice
    plt.contourf(x, y, data)
    contours = plt.contour(
        x, y, data, colors="white", linewidths=0.5, alpha=0.6
    )
    plt.clabel(contours)
    plt.title(title or var_name)
    if np.isclose(slice_extent[0], slice_extent[1]):
        plt.gca().set_aspect("equal")
    plt.xlabel(x_label)
    plt.ylabel(y_label)


if __name__ == "__main__":
    plot_slice_command(help_option_names=["-h", "--help"])
