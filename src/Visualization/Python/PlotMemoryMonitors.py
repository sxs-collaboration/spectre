#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.
import logging
import os
from math import inf
from typing import Optional, Sequence

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe

logger = logging.getLogger(__name__)


@click.command(name="memory-monitors")
@click.argument(
    "reduction_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
)
# Plotting options
@click.option(
    "--use-mb/--use-gb",
    default=False,
    show_default=True,
    help="Plot the y-axis in Megabytes or Gigabytes",
)
@click.option(
    "--x-label",
    help="The label on the x-axis.",
    show_default="name of the x-axis column",
)
@click.option(
    "--x-bounds",
    type=float,
    nargs=2,
    help="The lower and upper bounds of the x-axis.",
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_memory_monitors_command(
    reduction_files: Sequence[str],
    use_mb: bool,
    x_label: Optional[str],
    x_bounds: Optional[Sequence[float]],
):
    """
    Plot the memory usage of a simulation from the MemoryMonitors data in the
    Reductions H5 file.

    This tool assumes there is a group in each of the "reduction-files" with the
    path "/MemoryMonitors/" that holds dat files for each parallel component
    that was monitored.

    Note that the totals plotted here are not necessary the total memory usage
    of the simulation. The memory monitors only capture what is inside
    'pup'functions. Any memory that cannot be captured by a 'pup' function will
    not by represented by this plot.
    """

    # Given an h5 file, make sure that the "MemoryMonitors" group exists. Then
    # return the list of subfiles in that group
    def check_memory_monitor_dir(h5_filename: str):
        h5file = h5py.File(h5_filename, "r")
        memory_monitor_dir = h5file.get("MemoryMonitors")
        if memory_monitor_dir is None:
            raise click.UsageError(
                "Unable to open group 'MemoryMonitors' from h5 file"
                f" {h5_filename}. Available subfiles are:\n"
                f" {available_subfiles(h5file, extension='.dat')}"
            )

        return list(memory_monitor_dir.keys())

    # Given an h5 file and a subfile, make sure that the subfile exists inside
    # the h5 file. Then return the subfile as a DataFrame (with "Time" as the
    # index)
    def check_memory_monitor_file(
        h5_filename: str, h5file: h5py.File, subfile_name: str
    ):
        subfile_path = f"/MemoryMonitors/{subfile_name}"
        subfile = h5file.get(subfile_path)
        if subfile_path is None:
            raise click.UsageError(
                f"Unable to open memory file '{subfile_path}'"
                f" from h5 file {h5_filename}. Available subfiles are:\n"
                f" {available_subfiles(h5file, extension='.dat')}"
            )

        return to_dataframe(subfile).set_index("Time")

    # Given a DataFrame, sum all the columns that list totals for each node
    def total_over_nodes(df: pd.DataFrame):
        cols_to_sum = [
            col
            for col in df.columns
            if col == "Size (MB)" or "Size on node" in col
        ]
        return df[cols_to_sum].sum(axis=1)

    # Get a list of all components that we have monitored from the first
    # reductions file
    memory_filenames = check_memory_monitor_dir(reduction_files[0])

    # Open every h5file. For each h5file, turn each subfile into a DataFrame.
    # Then concat all DataFrames together into one that's indexed by the
    # subfile/component name.
    totals_df = pd.DataFrame()
    for reduction_file in reduction_files:
        h5file = h5py.File(reduction_file)
        local_totals_df = pd.DataFrame()

        for subfile_name in memory_filenames:
            df = check_memory_monitor_file(reduction_file, h5file, subfile_name)
            local_totals_df[subfile_name] = total_over_nodes(df)

        totals_df = pd.concat([totals_df, local_totals_df])

    # Restrict to x-bounds if there are any
    for subfile in memory_filenames:
        if x_bounds is not None:
            totals_df[subfile] = totals_df[subfile][
                (totals_df.index >= x_bounds[0])
                & (totals_df.index <= x_bounds[1])
            ]

    # Need .dat because all other components have that extension
    the_rest_str = "The Rest.dat"

    # Sum all the columns that list totals for each node to get a grand total
    # memory usage. Also see if any columns are less than 1% of this total. If
    # they are, group them in "the rest".
    grand_total = totals_df.sum(axis=1)
    the_rest_threshold = 0.01 * grand_total.mean()
    components_to_plot = list(
        totals_df.columns[totals_df.max() > the_rest_threshold]
    )
    the_rest_columns = list(set(totals_df.columns) - set(components_to_plot))
    # Only plot the rest of we actually have it
    if len(the_rest_columns) != 0:
        totals_df[the_rest_str] = totals_df[the_rest_columns].sum(axis=1)
        components_to_plot.append(the_rest_str)

    # For plotting in MB vs GB
    divisor = 1.0 if use_mb else 1000.0

    # Start plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the total
    ax.plot(
        totals_df.index,
        grand_total / divisor,
        color="black",
        label="Total",
    )

    # Determine plotting order
    maxes = list(totals_df[components_to_plot].max())
    components_to_plot = np.array(components_to_plot)[np.argsort(maxes)[::-1]]

    # Plot the individual components
    for component in components_to_plot:
        ax.plot(
            totals_df.index,
            totals_df[component] / divisor,
            linewidth=0.2,
            # Remove .dat extension
            label=component[:-4],
        )

    gb_or_mb = "MB" if use_mb else "GB"
    plt.title(f"Total Memory Usage ({gb_or_mb})")
    if x_label is not None:
        ax.set_xlabel(x_label)
    # The lines in the legend are a bit small because of the plot linewidth,
    # so make the legend lines a bit bigger
    leg = plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    for line in leg.get_lines():
        line.set_linewidth(1.0)


if __name__ == "__main__":
    plot_memory_monitors_command(help_option_names=["-h", "--help"])
