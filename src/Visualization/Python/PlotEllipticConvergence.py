#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import datetime
import logging
from typing import List

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe

logger = logging.getLogger(__name__)


def split_iteration_sequence(data: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a dataframe where its index is not strictly increasing.

    Returns a list of dataframes, where each has a strictly increasing index.
    """
    split_indices = np.split(
        np.arange(len(data)), np.nonzero(np.diff(data.index) < 1)[0] + 1
    )
    return [data.iloc[split_index] for split_index in split_indices]


def plot_elliptic_convergence(
    h5_file,
    fig=None,
    linear_residuals_subfile_name="GmresResiduals.dat",
    nonlinear_residuals_subfile_name="NewtonRaphsonResiduals.dat",
):
    """Plot elliptic solver convergence

    Arguments:
      h5_file: The H5 reductions file.
      fig: Optional. The matplotlib figure to plot in.
      linear_residuals_subfile_name: The name of the subfile containing the
        linear solver residuals.
      nonlinear_residuals_subfile_name: The name of the subfile containing the
        nonlinear solver residuals.
    """
    with h5py.File(h5_file, "r") as open_h5file:
        if not linear_residuals_subfile_name in open_h5file:
            all_subfiles = available_subfiles(open_h5file, extension=".dat")
            raise ValueError(
                "Could not find the linear residuals subfile"
                f" '{linear_residuals_subfile_name}' in the H5 file. Available"
                f" subfiles: {all_subfiles}"
            )
        linear_residuals = split_iteration_sequence(
            to_dataframe(open_h5file[linear_residuals_subfile_name]).set_index(
                "Iteration"
            )
        )
        nonlinear_residuals = (
            split_iteration_sequence(
                to_dataframe(
                    open_h5file[nonlinear_residuals_subfile_name]
                ).set_index("Iteration")
            )
            if nonlinear_residuals_subfile_name in open_h5file
            else None
        )
    cumulative_linsolv_iterations = [0] + list(
        np.cumsum([len(l) - 1 for l in linear_residuals])
    )
    norm = (
        nonlinear_residuals
        if nonlinear_residuals is not None
        else linear_residuals
    )[0]["Residual"].iloc[0]
    # Plot nonlinear solver residuals
    if fig is None:
        fig = plt.figure()
    ax_residual, ax_time = fig.subplots(
        nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    if nonlinear_residuals is not None:
        m = 0
        for i, residuals in enumerate(nonlinear_residuals):
            ax_residual.plot(
                cumulative_linsolv_iterations[m : m + len(residuals)],
                residuals["Residual"] / norm,
                color="black",
                ls="dotted",
                marker=".",
                label="Nonlinear solver" if i == 0 else None,
            )
            if "Walltime" in residuals:
                ax_time.plot(
                    cumulative_linsolv_iterations[m : m + len(residuals) - 1],
                    np.diff(residuals["Walltime"]),
                    color="black",
                    ls="dotted",
                    marker=".",
                )
            m += len(residuals) - 1
    # Plot linear solver residuals
    for i, residuals in enumerate(linear_residuals):
        ax_residual.plot(
            residuals.index + cumulative_linsolv_iterations[i],
            residuals["Residual"] / norm,
            color="black",
            label="Linear solver" if i == 0 else None,
            marker="." if len(residuals) < 20 else None,
        )
        if "Walltime" in residuals:
            ax_time.plot(
                residuals.index[:-1] + cumulative_linsolv_iterations[i],
                np.diff(residuals["Walltime"]),
                color="black",
                marker="." if len(residuals) < 20 else None,
            )
    # Annotate time of last linear solver iteration
    if "Walltime" in residuals:
        last_time = linear_residuals[-1].iloc[-1]["Walltime"]
        ax_time.annotate(
            f"Walltime: {datetime.timedelta(seconds=round(last_time))}",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
        )
    else:
        ax_time.annotate(
            "No walltime data",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
        )

    # Configure the axes
    ax_residual.set_yscale("log")
    ax_residual.grid()
    ax_residual.legend()
    ax_residual.set_ylabel("Relative residual")
    ax_time.set_yscale("log")
    ax_time.grid()
    ax_time.set_ylabel("Walltime per iteration [s]")
    ax_time.set_xlabel("Cumulative linear solver iteration")
    # Allow only integer ticks for the x-axis
    ax_time.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


@click.command(name="elliptic-convergence")
@click.argument(
    "h5_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--linear-residuals-subfile-name",
    help="The name of the subfile containing the linear solver residuals",
    default="GmresResiduals.dat",
    show_default=True,
)
@click.option(
    "--nonlinear-residuals-subfile-name",
    help="The name of the subfile containing the nonlinear solver residuals",
    default="NewtonRaphsonResiduals.dat",
    show_default=True,
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_elliptic_convergence_command(**kwargs):
    """Plot elliptic solver convergence"""
    _rich_traceback_guard = True  # Hide traceback until here
    return plot_elliptic_convergence(**kwargs)


if __name__ == "__main__":
    plot_elliptic_convergence_command(help_option_names=["-h", "--help"])
