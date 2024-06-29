# Distributed under the MIT License.
# See LICENSE.txt for details.
# Plot trajectory from inspiral run, concatenating all segments

import logging

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)


def import_A_and_B(filenames, subfile_name_aha, subfile_name_ahb):
    A_data = []
    B_data = []

    for filename in filenames:
        with h5py.File(filename, "r") as file:
            A_data.append(
                np.array(
                    file[subfile_name_aha][:, [4, 5, 6]]
                )  # 0 ->time,  4 -> x, 5 -> y, 6 -> z
            )
            B_data.append(
                np.array(
                    file[subfile_name_ahb][:, [4, 5, 6]]
                )  # 0 ->time,  4 -> x, 5 -> y, 6 -> z
            )

    if A_data and B_data:
        A = np.concatenate(A_data)
        B = np.concatenate(B_data)
        min_length = min(len(A), len(B))
        return A[:min_length], B[:min_length]
    else:
        return None, None


def plot_trajectory(AhA: np.ndarray, AhB: np.ndarray, fig=None):
    """
    Plot concatenated trajectories in inspiral simulation.

    Outputs a 2x2 figure with 4 plots:
    1st row: A and B trajectories in 3D, coordinate separation in 3D
    2nd row: A and B trajectories in 2D xy plane, coordinate separation in 2D
    xy plane. The 3D plots use a subsampled version of the data, with default
    value of 15, to speed up the plots and avoid memory error.

    Arguments:
    AhA: Array of shape (num_points, 3) with the coordinates of
    the first object.
    AhB: Array of shape (num_points, 3) with the coordinates of
    the second object.
    fig: Matplotlib figure object used for plotting. Subplots will be added to
      this figure. If None, a new figure will be created.
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 10))

    # Plot 3D trajectories
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(*AhA.T, color="C0", label="AhA")
    ax1.scatter(*AhA[-1], color="C0")
    ax1.plot(*AhB.T, color="C1", label="AhB")
    ax1.scatter(*AhB[-1], color="C1")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Trajectories")
    ax1.legend()

    # Calculate coordinate separation in 3D
    separation_3d = AhA - AhB

    # Plot 3D coordinate separation
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.plot(*separation_3d.T, color="black")
    ax2.scatter(*separation_3d[-1], color="black")
    ax2.set_xlabel("X diff")
    ax2.set_ylabel("Y diff")
    ax2.set_zlabel("Z diff")
    ax2.set_title("Coordinate separation (3D)")

    # Plot 2D trajectories
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(*AhA[:, 0:2].T, label="AhA", color="C0")
    ax3.scatter(*AhA[-1, 0:2], color="C0")
    ax3.plot(*AhB[:, 0:2].T, label="AhB", color="C1")
    ax3.scatter(*AhB[-1, 0:2], color="C1")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.set_title("Trajectories (2D)")
    ax3.set_aspect("equal")  # Set aspect ratio to 1
    ax3.grid(True)  # Add gridlines

    # Calculate coordinate separation in 2D
    separation_2d = AhA[:, 0:2] - AhB[:, 0:2]
    # Plot coordinate separation in 2D
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(*separation_2d.T, color="black")
    ax4.scatter(*separation_2d[-1], color="black")
    ax4.set_xlabel("x1 - x2")
    ax4.set_ylabel("y1 - y2")
    ax4.set_title("Coordinate separation (2D)")
    ax4.set_aspect("equal")  # Set aspect ratio to 1
    ax4.grid(True)  # Add gridlines

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return fig


@click.command(name="trajectories")
@click.argument(
    "h5_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name-aha",
    "-A",
    default="ApparentHorizons/ControlSystemAhA_Centers.dat",
    help=(
        "Name of subfile containing the apparent horizon centers for object A."
    ),
)
@click.option(
    "--subfile-name-ahb",
    "-B",
    default="ApparentHorizons/ControlSystemAhB_Centers.dat",
    help=(
        "Name of subfile containing the apparent horizon centers for object B."
    ),
)
@click.option(
    "--sample-rate",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Downsample data to speed up plots. E.g. a value of 10 means every 10th"
        " point is plotted."
    ),
)
@click.option(
    "--figsize",
    nargs=2,
    type=float,
    default=(10.0, 10.0),
    show_default=True,
    help="Figure size as width and height in inches",
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_trajectories_command(
    h5_files, subfile_name_aha, subfile_name_ahb, sample_rate, figsize
):
    """Plot trajectories in inspiral simulation

    Concatenates partial trajectories from each h5 reductions files and plots
    full trajectories.
    """
    AhA, AhB = import_A_and_B(h5_files, subfile_name_aha, subfile_name_ahb)
    fig = plt.figure(figsize=figsize)
    return plot_trajectory(AhA[::sample_rate], AhB[::sample_rate], fig)


if __name__ == "__main__":
    plot_trajectories_command(help_option_names=["-h", "--help"])
