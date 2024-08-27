#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Optional, Sequence

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectre.support.CliExceptions import RequiredChoiceError
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe

logger = logging.getLogger(__name__)


def plot_control_system(
    reduction_files: Sequence[str],
    with_shape: bool = True,
    show_all_m: bool = False,
    shape_l_max: int = 2,
    x_bounds: Optional[Sequence[float]] = None,
    x_label: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot diagnostic information regarding all control systems except size
    control. If you want size control diagnostics use
    `spectre plot size-control`.

    This tool assumes there are subfiles in each of the "reduction-files" with
    the path `/ControlSystems/{Name}/*.dat`, where `{NAME}` is the name of the
    control system and `*.dat` are all the components of that control system.

    Shape control is a bit special because it has a large number of components.
    Control whether or not you plot shape, and how many of these components you
    plot, with the `--with-shape/--without-shape`, `--shape-l_max`, and
    `--show-all-m` options.
    """

    # Given an h5 file, make sure that the "ControlSystems" group exists. Then
    # return the list of control systems in that group, excluding size control
    def check_control_system_dir(h5_filename: str):
        h5file = h5py.File(h5_filename, "r")
        control_system_dir = h5file.get("ControlSystems")
        if control_system_dir is None:
            raise RequiredChoiceError(
                (
                    "Unable to open group 'ControlSystems' from h5 file"
                    f" {h5_filename}."
                ),
                choices=available_subfiles(h5file, extension=".dat"),
            )

        # No size control
        return [key for key in control_system_dir.keys() if "Size" not in key]

    # Given an h5 file and a subfile, make sure that the subfile exists inside
    # the h5 file. Then return the subfile as a DataFrame (with "Time" as the
    # index)
    def check_control_system_file(
        h5_filename: str,
        h5file: h5py.File,
        control_system_name: str,
        component_name: str,
    ):
        subfile_path = (
            f"/ControlSystems/{control_system_name}/{component_name}.dat"
        )
        subfile = h5file.get(subfile_path)
        if subfile_path is None:
            raise RequiredChoiceError(
                (
                    f"Unable to open control system subfile '{subfile_path}'"
                    f" from h5 file {h5_filename}."
                ),
                choices=available_subfiles(h5file, extension=".dat"),
            )

        return to_dataframe(subfile).set_index("Time")

    # Given an h5 file name and a control system name, return a list of all
    # components for that control system
    def get_control_system_components(
        h5_filename: str, control_system_name: str
    ):
        h5file = h5py.File(h5_filename, "r")
        control_system_group_name = f"/ControlSystems/{control_system_name}"
        control_system_group = h5file.get(control_system_group_name)

        # Only want the component name without the '.dat' extension
        keys = [key.split("/")[-1][:-4] for key in control_system_group.keys()]

        if "Shape" in control_system_name:
            return [
                key
                for key in keys
                if 2 <= int(key.split("l")[1].split("m")[0]) <= shape_l_max
            ]
        else:
            return keys

    relevant_columns = ["ControlError", "DampingTimescale"]

    # We are only plotting the most relevant columns for now. More can be added
    # later if it's useful for debugging
    def extract_relevant_columns(
        df: pd.DataFrame, name: str, component_name: str
    ):
        return df[relevant_columns].add_prefix(name + component_name)

    # Get a list of all control systems (excluding size) that we have from the
    # first reductions file and a map to all of their components
    control_systems = check_control_system_dir(reduction_files[0])
    control_system_components = {
        system: get_control_system_components(reduction_files[0], system)
        for system in control_systems
        if "Shape" not in system or with_shape
    }

    # Open every h5file. For each h5file, turn each component of each control
    # system into a DataFrame and concat it with the large data frame
    data = pd.DataFrame()
    for reduction_file in reduction_files:
        h5file = h5py.File(reduction_file)
        file_df = pd.DataFrame()

        for system in control_system_components:
            for component in control_system_components[system]:
                tmp_data = check_control_system_file(
                    reduction_file, h5file, system, component
                )
                tmp_data = extract_relevant_columns(tmp_data, system, component)
                # When we concat DataFrames from within an H5 file together, we
                # assume they have the same indexes (times) so we concat along
                # axis=1
                file_df = pd.concat([file_df, tmp_data], axis=1)

            if "Shape" not in system or show_all_m:
                continue

            # If this is the shape system and we don't want to show all m, we
            # take the L2 norm over all m for a given l
            for l in range(2, shape_l_max + 1):
                component_prefix = f"l{l}m"
                for column in relevant_columns:
                    components_to_norm = [
                        f"{system}{component_prefix}{m}{column}"
                        for m in range(-l, l + 1)
                    ]

                    file_df[f"{system}{component_prefix}{column}"] = np.sqrt(
                        np.square(file_df[components_to_norm].to_numpy()).sum(
                            axis=1
                        )
                    )

        # When concating the large DataFrames from each H5 file together, we
        # assume all the columns are the same so we concat along axis=0
        data = pd.concat([data, file_df])

    # If we aren't showing all m for shape control, modify the shape components
    # that are being plotted
    if with_shape and not show_all_m:
        for system in control_systems:
            if "Shape" not in system:
                continue

            # Overwrite existing components with empty list
            control_system_components[system] = []
            for l in range(2, shape_l_max + 1):
                control_system_components[system].append(f"l{l}m")

    # Restrict data to bounds
    if x_bounds:
        data = data[(data.index >= x_bounds[0]) & (data.index <= x_bounds[1])]

    # Set up plots
    fig, axes = plt.subplots(2, 1, sharex=True)

    axes = list(axes)

    # Plot all components for all systems. The name for the legend differs
    # slightly from the name in the DataFrame, just to make it prettier for the
    # user. Since both figures will have the same lines, we only show one legend
    for system in control_system_components:
        for component in control_system_components[system]:
            df_name = f"{system}{component}"
            legend_name = (
                system
                if len(control_system_components[system]) == 1
                else f"{system} {component}"
            )

            axes[0].plot(
                data.index,
                np.abs(data[f"{df_name}ControlError"]),
            )
            axes[1].plot(
                data.index,
                np.abs(data[f"{df_name}DampingTimescale"]),
                label=legend_name,
            )

    # Configure the axes and legend
    fig.set_size_inches(10, len(axes) * 4)
    axes[0].set_ylabel("Control Error")
    axes[1].set_ylabel("Damping Time")
    for ax in axes:
        ax.grid()
        ax.set_yscale("log")
    if x_label:
        # Bottom plot gets x label
        axes[1].set_xlabel(x_label)
    if title:
        fig.suptitle(title)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 1.1))
    return fig


@click.command(name="control-system", help=plot_control_system.__doc__)
@click.argument(
    "reduction_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--with-shape/--without-shape",
    default=True,
    show_default=True,
    help="Wether or not to plot shape control.",
)
@click.option(
    "--shape-l_max",
    "-l",
    type=int,
    default=2,
    show_default=True,
    help=(
        "The max number of spherical harmonics to show on the plot. Since"
        " higher ell can have a lot of components, it may be desirable to show"
        " fewer components. Never plots l=0,1 since we don't control these"
        " components. Only used if '--with-shape'."
    ),
)
@click.option(
    "--show-all-m",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "When plotting shape control, for a given ell, plot all m components."
        " Default is, for a given ell, to plot the L2 norm over all the m"
        " components. Only used if '--with-shape'."
    ),
)
# Plotting options
@click.option(
    "--x-bounds",
    type=float,
    nargs=2,
    help="The lower and upper bounds of the x-axis.",
)
@click.option(
    "--x-label",
    help="The label on the x-axis.",
)
@click.option(
    "--title",
    "-t",
    help="Title of the graph.",
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_control_system_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    return plot_control_system(**kwargs)


if __name__ == "__main__":
    plot_control_system_command(help_option_names=["-h", "--help"])
