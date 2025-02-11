#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Sequence, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm
from scipy.interpolate import CloughTocher2DInterpolator

from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)

logger = logging.getLogger(__name__)


def plot_eccentricity_control(
    ecc_params_files: Sequence[Union[str, Path]],
):
    """Plot eccentricity control iterations.

    \f
    Arguments:
      ecc_params_files: YAML files containing the output of
        'eccentricity_control_params()'.

    Returns:
      The figure containing the plot.
    """
    # Load the eccentricity parameters
    ecc_params = []
    for ecc_params_file in ecc_params_files:
        with open(ecc_params_file, "r") as open_file:
            ecc_params.append(yaml.safe_load(open_file))
    ecc_params = pd.DataFrame(ecc_params)

    # Set up the figure
    fig = plt.figure(figsize=(4.5, 4))
    plt.xlabel(r"Angular velocity $\Omega_0$")
    plt.ylabel(r"Expansion $\dot{a}_0$")
    plt.grid(zorder=0)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot measured eccentricity contours
    ecc_norm = LogNorm(vmin=1e-4, vmax=1)
    if len(ecc_params) > 2:
        ecc_intrp = CloughTocher2DInterpolator(
            ecc_params[["Omega0", "Adot0"]],
            np.log10(ecc_params["Eccentricity"]),
            rescale=True,
        )
        x, y = np.meshgrid(
            np.linspace(*ecc_params["Omega0"].agg(["min", "max"]), 200),
            np.linspace(*ecc_params["Adot0"].agg(["min", "max"]), 200),
        )
        z = ecc_intrp(x, y)
        ecc_contours = plt.contourf(
            x,
            y,
            10**z,
            levels=10.0 ** np.arange(-4, 1),
            norm=ecc_norm,
            cmap="Blues",
            zorder=10,
        )
    else:
        ecc_contours = plt.cm.ScalarMappable(
            norm=ecc_norm,
            cmap="Blues",
        )
    plt.colorbar(ecc_contours, label="Eccentricity", ax=plt.gca())
    cbar_ax = fig.axes[1]
    plt.gca().use_sticky_edges = False

    # Plot the path through parameter space
    plt.plot(
        ecc_params["Omega0"],
        ecc_params["Adot0"],
        color="black",
        zorder=20,
    )

    # Place a marker at each iteration
    scatter_kwargs = dict(
        color="white",
        marker="o",
        linewidth=1,
        edgecolor="black",
        s=100,
        zorder=25,
    )
    annotate_kwargs = dict(
        textcoords="offset points",
        xytext=(0, -0.5),
        ha="center",
        va="center",
        fontsize=8,
        zorder=30,
    )
    plt.scatter(ecc_params["Omega0"], ecc_params["Adot0"], **scatter_kwargs)
    cbar_ax.scatter(
        np.repeat(0.5, len(ecc_params)),
        ecc_params["Eccentricity"],
        **scatter_kwargs,
    )
    for i in range(len(ecc_params)):
        plt.annotate(
            f"{i}",
            ecc_params.iloc[i][["Omega0", "Adot0"]],
            **annotate_kwargs,
        )
        cbar_ax.annotate(
            f"{i}", (0.5, ecc_params["Eccentricity"][i]), **annotate_kwargs
        )

    # Plot location of the next iteration
    plt.plot(
        ecc_params.iloc[-1][["Omega0", "NewOmega0"]],
        ecc_params.iloc[-1][["Adot0", "NewAdot0"]],
        color="black",
        linestyle="dotted",
        zorder=20,
    )
    plt.scatter(
        ecc_params.iloc[-1]["NewOmega0"],
        ecc_params.iloc[-1]["NewAdot0"],
        color="black",
        marker="o",
        s=30,
        zorder=40,
    )
    return fig


@click.command(
    name="eccentricity-control", help=plot_eccentricity_control.__doc__
)
@click.argument(
    "ecc_params_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_eccentricity_control_command(**kwargs):
    return plot_eccentricity_control(**kwargs)


if __name__ == "__main__":
    plot_eccentricity_control_command(help_option_names=["-h", "--help"])
