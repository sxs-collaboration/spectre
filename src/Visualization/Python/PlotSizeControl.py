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


@click.command(name="size-control")
@click.argument(
    "reduction_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--object-label",
    "-d",
    required=True,
    help=(
        "Which object to plot. This is either 'A', 'B', or 'None'. 'None' is"
        " used when there is only one black hole in the simulation."
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
def plot_size_control_command(
    reduction_files: Sequence[str],
    object_label: str,
    x_bounds: Optional[Sequence[float]],
    x_label: Optional[str],
    title: Optional[str],
):
    """
    Plot diagnostic information regarding the Size control system.

    This tool assumes there is a subfile in each of the "reduction-files" with
    the path `/ControlSystems/Size{LABEL}/Diagnostics.dat`, where `{LABEL}` is
    replaced with the "object-label" input option.
    """
    object_label = (
        "" if object_label.lower() == "none" else object_label.upper()
    )

    def check_diagnostics_file(h5_filename):
        with h5py.File(h5_filename, "r") as h5file:
            # See src/ControlSystem/ControlErrors/Size.cpp for path
            diagnostics_file_name = (
                f"ControlSystems/Size{object_label}/Diagnostics.dat"
            )
            diagnostics_data = h5file.get(diagnostics_file_name)
            if diagnostics_data is None:
                raise RequiredChoiceError(
                    (
                        "Unable to open diagnostic file"
                        f" '{diagnostics_file_name}' from h5 file"
                        f" {h5_filename}."
                    ),
                    choices=available_subfiles(h5file, extension=".dat"),
                )

            return to_dataframe(diagnostics_data)

    data = pd.concat(
        check_diagnostics_file(reduction_file)
        for reduction_file in reduction_files
    )

    # Restrict data to bounds
    if x_bounds:
        data = data[
            (data["Time"] >= x_bounds[0]) & (data["Time"] <= x_bounds[1])
        ]

    # Set up plots
    fig, axes = plt.subplots(7, 1, sharex=True)

    # Plot the selected quantities. Groupings are based both on relationship
    # of quantities and also relative scales of quantities
    times = data["Time"]
    axes = list(axes)
    # Damping times are sort of the odd ones out. Put it with control error but
    # on a different y-axis
    top1 = axes[0].plot(
        times, np.abs(data["ControlError"]), label="Control Error"
    )
    axes.append(axes[0].twinx())
    top2 = axes[-1].plot(
        times, data["DampingTime"], color="C1", label="Damping time"
    )
    axes[0].set_yscale("log")
    axes[-1].set_yscale("log")
    # Support an older version of control system output that didn't have the
    # smoother timescale
    if "SmootherTimescale" in data.columns:
        top3 = axes[-1].plot(
            times,
            data["SmootherTimescale"],
            color="C1",
            linestyle="dashed",
            label="Smooth damping time",
        )
        top_lines = top1 + top2 + top3
    else:
        top_lines = top1 + top2

    top_labels = [l.get_label() for l in top_lines]
    axes[0].tick_params(axis="y", labelcolor="C0")
    axes[-1].tick_params(axis="y", labelcolor="C1")

    # State
    axes[1].plot(times, data["StateNumber"], label="State")
    axes[1].set_ylim(0, 5)

    # All delta R's
    axes[2].plot(
        times,
        data["MinRelativeDeltaR"],
        label="Min relative Delta R",
        color="C0",
    )
    axes[2].plot(
        times,
        data["AvgRelativeDeltaR"],
        label="Average relative Delta R",
        color="C1",
    )
    axes[2].plot(
        times,
        data["MinDeltaR"],
        label="Min Delta R",
        color="C0",
        linestyle="dashed",
    )
    axes[2].plot(
        times,
        data["AvgDeltaR"],
        label="Average Delta R",
        color="C1",
        linestyle="dashed",
    )

    # Char speeds
    axes[3].plot(times, data["MinCharSpeed"], label="Min char speed")
    axes[3].plot(
        times, data["MinComovingCharSpeed"], label="Min comoving char speed"
    )
    axes[3].plot(times, data["TargetCharSpeed"], label="Target char speed")

    # Zero crossing predictors
    axes[4].plot(
        times, data["CharSpeedCrossingTime"], label="Char speed crossing time"
    )
    axes[4].plot(
        times,
        data["ComovingCharSpeedCrossingTime"],
        label="Comoving char speed crossing time",
    )
    axes[4].plot(
        times, data["DeltaRCrossingTime"], label="Delta R crossing time"
    )
    axes[4].set_yscale("log")

    # Lambda and horizon
    axes[5].plot(times, data["FunctionOfTime"], label="Function of time")
    axes[5].plot(times, data["HorizonCoef00"], label="Horizon coef 00")

    # Time deriv of lambda and horizon
    axes[6].plot(times, data["DtFunctionOfTime"], label="dtFunction of time")
    axes[6].plot(
        times, data["RawDtHorizonCoef00"], label="Raw dtHorizon coef 00"
    )
    axes[6].plot(
        times, data["AveragedDtHorizonCoef00"], label="Avg dtHorizon coef 00"
    )

    # Configure the axes and legends
    fig.set_size_inches(10, len(axes) * 2)
    if x_label:
        # Bottom plot. -1 is the second y-axis for top plot
        axes[-2].set_xlabel(x_label)
    if title:
        plt.title(title)
    for i in range(len(axes) - 1):
        axes[i].grid()
        if i == 0:
            axes[i].legend(
                top_lines,
                top_labels,
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
            )
        else:
            axes[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))


if __name__ == "__main__":
    plot_size_control_command(help_option_names=["-h", "--help"])
