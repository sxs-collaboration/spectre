#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

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
from spectre.Visualization.ReadH5 import to_dataframe

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
    ax=None,
    linear_residuals_subfile_name="GmresResiduals.dat",
    nonlinear_residuals_subfile_name="NewtonRaphsonResiduals.dat",
):
    """Plot elliptic solver convergence

    Arguments:
      h5_file: The H5 reductions file.
      ax: Optional. The matplotlib axis to plot on.
      linear_residuals_subfile_name: The name of the subfile containing the
        linear solver residuals.
      nonlinear_residuals_subfile_name: The name of the subfile containing the
        nonlinear solver residuals.
    """
    with h5py.File(h5_file, "r") as open_h5file:
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
    if ax is not None:
        plt.sca(ax)
    if nonlinear_residuals is not None:
        m = 0
        for i, residuals in enumerate(nonlinear_residuals):
            plt.plot(
                cumulative_linsolv_iterations[m : m + len(residuals)],
                residuals["Residual"] / norm,
                color="black",
                ls="dotted",
                marker=".",
                label="Nonlinear residual" if i == 0 else None,
            )
            m += len(residuals) - 1
    # Plot linear solver residuals
    for i, residuals in enumerate(linear_residuals):
        plt.plot(
            residuals.index + cumulative_linsolv_iterations[i],
            residuals["Residual"] / norm,
            color="black",
            label="Linear residual" if i == 0 else None,
            marker="." if len(residuals) < 20 else None,
        )

    # Configure the axes
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("Cumulative linear solver iteration")
    plt.ylabel("Relative residual")
    plt.title("Elliptic solver convergence")
    # Allow only integer ticks for the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


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
    plot_elliptic_convergence(**kwargs)


if __name__ == "__main__":
    plot_elliptic_convergence_command(help_option_names=["-h", "--help"])
