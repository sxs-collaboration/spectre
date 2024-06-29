#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Optional, Sequence

import click
import h5py
import matplotlib.pyplot as plt
import pandas as pd

from spectre.support.CliExceptions import RequiredChoiceError
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from spectre.Visualization.ReadH5 import available_subfiles, to_dataframe

logger = logging.getLogger(__name__)


def _parse_modes(ctx, param, all_modes):
    result = []
    for mode in all_modes:
        l_and_m = mode.split(",")
        if len(l_and_m) != 2:
            raise click.BadParameter(
                f"The mode {mode} must be specified as 'l,m'"
            )
        result.append(f"Real Y_{mode}")
        result.append(f"Imag Y_{mode}")
    return result


def plot_cce(
    h5_filename: str,
    modes: Sequence[str],
    real: bool = False,
    imag: bool = False,
    extraction_radius: Optional[int] = None,
    list_extraction_radii: bool = False,
    backward_cce_group: Optional[str] = None,
    x_bounds: Optional[Sequence[float]] = None,
    x_label: Optional[str] = None,
    title: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
):
    """
    Plot the Strain, News, and Psi0-Psi4 from the output of a SpECTRE CCE run.

    The data must be in a SpECTRE Cce subfile with a '.cce' extension. Multiple
    modes can be plotted at once along with your choice of plotting both real,
    imaginary, or both.

    IMPORTANT: These plots are *NOT* in the correct BMS frame. This tool is only
    meant to plot the raw data produced by the SpECTRE CCE module.
    """

    if real and imag:
        raise click.UsageError(
            "Only specify one of '--real'/'--imag'. If you want to plot both"
            " real and imaginary modes, specify neither."
        )

    # Filter out modes we aren't plotting
    if real:
        modes = [mode for mode in modes if "Real" in mode]
    elif imag:
        modes = [mode for mode in modes if "Imag" in mode]

    plot_quantities = ["Strain", "News", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4"]

    with h5py.File(h5_filename, "r") as h5file:
        cce_subfiles = available_subfiles(h5file, extension=".cce")

        # If we're only listing subfiles, print them and exit
        if list_extraction_radii:
            import rich
            import rich.columns

            rich.print(rich.columns.Columns(cce_subfiles))
            return

        # If there aren't any cce subfiles, raise an error (unless this is the
        # old CCE data format)
        if len(cce_subfiles) == 0 and backward_cce_group is None:
            raise click.UsageError(
                f"Could not find any Cce subfiles in H5 file {h5_filename}. Cce"
                " subfiles must end with the extension '.cce'"
            )

        if (
            len(cce_subfiles) == 1
            and extraction_radius is not None
            and f"SpectreR{extraction_radius:04}.cce" not in cce_subfiles[0]
        ):
            raise click.UsageError(
                f"The extraction radius passed in ({extraction_radius}) does"
                " not match the single Cce subfile that was found"
                f" ({cce_subfiles[0]}). Either specify the correct extraction"
                " radius, or remove the option altogether."
            )

        if backward_cce_group is not None:
            cce_subfiles = [backward_cce_group]

        # If there is more than one cce subfile, but we didn't specify an
        # extraction radius, then error
        if len(cce_subfiles) > 1 and extraction_radius is None:
            raise click.UsageError(
                f"The H5 file {h5_filename} has {len(cce_subfiles)} Cce"
                " subfiles, but you did not specify an extraction radius."
                " Please specify an extraction radius with"
                " '--extraction-radius'/'-r'."
            )

        # If we didn't specify an extraction radius, the subfile name is just
        # the one listed in the file. If the extraction radius was specified
        # (and we've now guaranteed there is more than one subfile) use that as
        # the subfile name
        cce_subfile_name = (
            cce_subfiles[0]
            if (extraction_radius is None or backward_cce_group is not None)
            else f"SpectreR{extraction_radius:04}.cce"
        )
        cce_subfile = h5file.get(cce_subfile_name)
        if cce_subfile is None:
            raise RequiredChoiceError(
                (
                    f"Could not find Cce subfile {cce_subfile} in H5 file"
                    f" {h5_filename}."
                ),
                choices=cce_subfiles,
            )

        suffix = ".dat" if backward_cce_group is not None else ""

        # Only take the columns that we need and prefix the columns with their
        # quantity so we can have just one DataFrame
        data = pd.concat(
            [
                to_dataframe(cce_subfile.get(quantity + suffix))
                .set_index("time")[modes]
                .add_prefix(quantity)
                for quantity in plot_quantities
            ],
            axis=1,
        )

    # Restrict to x-bounds if we have any
    if x_bounds:
        data = data[(data.index >= x_bounds[0]) & (data.index <= x_bounds[1])]

    # Set up the plots
    if fig is None:
        fig = plt.figure(figsize=(8, 2 * len(plot_quantities)))
    axes = list(fig.subplots(len(plot_quantities), 1, sharex=True))
    # Make the legend look a bit nicer
    divisor = 1 if (real or imag) else 2
    num_col = min(4, len(modes) / divisor)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot quantities and configure the axes and legend
    for i, quantity in enumerate(plot_quantities):
        ax = axes[i]
        for j, mode in enumerate(modes):
            # If we are plotting both real and imaginary modes, make real solid
            # and imaginary dashed. If we are plotting only real or only
            # imaginary, then the lines are solid
            linestyle = (
                "solid" if (real or imag or "Real" in mode) else "dashed"
            )
            # If we are plotting both real and imaginary modes, make the
            # real/imag lines for the same mode the same color. If we are
            # plotting only real or only imaginary, then just cycle regularly
            # through the colors
            cycle_idx = (
                j % len(cycle) if (real or imag) else (j // 2) % len(cycle)
            )
            ax.plot(
                data.index,
                data[f"{quantity}{mode}"],
                color=cycle[cycle_idx],
                linestyle=linestyle,
                label=mode if i == 0 else None,
            )
        ax.set_ylabel(quantity)
        # Legend only above top plot
        if i == 0:
            ax.legend(
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=num_col
            )

    # If we have an x-label, goes on the bottom plot
    if x_label:
        axes[-1].set_xlabel(x_label)

    # Plot needs to be fairly big
    if title:
        plt.suptitle(title)
    return fig


@click.command(name="cce", help=plot_cce.__doc__)
@click.argument(
    "h5_filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=1,
)
@click.option(
    "--modes",
    "-m",
    multiple=True,
    callback=_parse_modes,
    required=True,
    help=(
        "Which mode to plot. Specified as 'l,m' (e.g. '--modes 2,2'). Will plot"
        " both real and imaginary components unless '--real' or '--imag' are"
        " specified. Can be specified multiple times."
    ),
)
@click.option(
    "--real",
    is_flag=True,
    default=False,
    help="Plot only real modes. Mutually exclusive with '--imag'.",
)
@click.option(
    "--imag",
    is_flag=True,
    default=False,
    help="Plot only imaginary modes. Mutually exclusive with '--real'.",
)
@click.option(
    "--extraction-radius",
    "-r",
    type=int,
    help=(
        "Extraction radius of data to plot as an int. If there is only one Cce"
        " subfile, that one will be used and this option does not need to be"
        " specified. The expected form of the Cce subfile is 'SpectreRXXXX.cce'"
        " where XXXX is the zero-padded integer extraction radius. This option"
        " is ignored if the backwards compatibility option '--cce-group'/'-d'"
        " is specified."
    ),
)
@click.option(
    "--list-extraction-radii",
    "-l",
    is_flag=True,
    default=False,
    help="List Cce subfiles in the 'h5_filename' and exit.",
)
@click.option(
    "--cce-group",
    "-d",
    "backward_cce_group",
    help=(
        "Option for backwards compatibility with an old version of CCE data."
        " This is the group name of the CCE data in the 'h5_filename'"
        " (typically Cce). This option should only be used if your CCE data was"
        " produced with a version of SpECTRE prior to this Pull Request:"
        " https://github.com/sxs-collaboration/spectre/pull/5985."
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
def plot_cce_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    plot_cce(**kwargs)


if __name__ == "__main__":
    plot_cce_command(help_option_names=["-h", "--help"])
