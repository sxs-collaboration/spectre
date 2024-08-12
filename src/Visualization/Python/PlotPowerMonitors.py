# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from itertools import cycle
from typing import Iterable, Optional, Sequence, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import Domain, deserialize_domain
from spectre.IO.H5.IterElements import iter_elements, stripped_element_name
from spectre.NumericalAlgorithms.LinearOperators import power_monitors
from spectre.Spectral import Basis
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.Visualization.OpenVolfiles import (
    open_volfiles,
    open_volfiles_command,
    parse_point,
)
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)

logger = logging.getLogger(__name__)


def find_block_or_group(
    block_id: int,
    block_or_group_names: Sequence[str],
    domain: Union[Domain[1], Domain[2], Domain[3]],
) -> Optional[int]:
    """Find entry in 'block_or_group_names' that corresponds to the 'block_id'"""
    block_name = domain.blocks[block_id].name
    for i, name in enumerate(block_or_group_names):
        if name == block_name:
            return i
        if (
            name in domain.block_groups
            and block_name in domain.block_groups[name]
        ):
            return i
    return None


def plot_power_monitors(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: Optional[int],
    tensor_components: Sequence[str],
    block_or_group_names: Sequence[str],
    domain: Union[Domain[1], Domain[2], Domain[3]],
    dimension_labels: Sequence[str] = [r"$\xi$", r"$\eta$", r"$\zeta$"],
    element_patterns: Optional[Sequence[str]] = None,
    skip_filtered_modes: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
):
    plot_over_time = obs_id is None
    # One column per block or group
    num_cols = len(block_or_group_names)
    # One row per dimension if plotted over time to declutter the plots
    num_rows = domain.dim if plot_over_time else 1
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=figsize or (num_cols * 4, num_rows * 4),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    # Evaluate property cycles (by default this is just 'color'). We do multiple
    # plotting commands (at least one per element), so we don't want matplotlib
    # to cycle through the properties at every plotting command.
    prop_cycle = {
        key: cycle(values)
        for key, values in plt.rcParams["axes.prop_cycle"].by_key().items()
    }
    props_dim = {
        d: {key: next(values) for key, values in prop_cycle.items()}
        for d in range(domain.dim)
    }

    # Collect data for each subplot
    if plot_over_time:
        all_mode_time_series = {
            subplot_index: dict() for subplot_index in range(num_cols)
        }
    else:
        num_elements = np.zeros(num_cols, dtype=int)
        max_error = np.zeros((num_cols, domain.dim))

    for element, tensor_data in iter_elements(
        volfiles, obs_id, tensor_components, element_patterns=element_patterns
    ):
        # Skip FD elements because we can't compute power monitors for them
        if any(
            basis == Basis.FiniteDifference for basis in element.mesh.basis()
        ):
            continue

        # Find the subplot for this element's block, or skip the element if its
        # block wasn't selected
        subplot_index = find_block_or_group(
            element.id.block_id, block_or_group_names, domain
        )
        if subplot_index is None:
            continue

        # Compute power monitors and take L2 norm over tensor components
        all_modes = [
            np.zeros(element.mesh.extents(d) - skip_filtered_modes)
            for d in range(element.dim)
        ]
        for component in tensor_data:
            modes = power_monitors(DataVector(component), element.mesh)
            for d, modes_dim in enumerate(modes):
                num_modes = len(modes_dim) - skip_filtered_modes
                all_modes[d] += np.array(modes_dim)[:num_modes] ** 2
        for d in range(element.dim):
            all_modes[d] = np.sqrt(all_modes[d])

        if plot_over_time:
            # Collect time series of modes
            all_mode_time_series[subplot_index].setdefault(
                element.id, []
            ).append((element.time, all_modes))
        else:
            # Plot modes directly
            ax = axes[0][subplot_index]
            for d, modes_dim in enumerate(all_modes):
                ax.semilogy(modes_dim, **props_dim[d], zorder=30 + d)
                ax.scatter(
                    len(modes_dim) - 1,
                    modes_dim[-1],
                    marker=".",
                    color=props_dim[d].get("color", "black"),
                    zorder=30 + d,
                )
                # Collect reduction data
                # - We estimate the truncation error by just taking the highest
                #   mode. This won't work well with filtering and should be
                #   improved on the C++ side.
                max_error[subplot_index][d] = max(
                    max_error[subplot_index][d], all_modes[d][-1]
                )
            num_elements[subplot_index] += 1

    if plot_over_time:
        # Plot mode timeseries
        max_num_modes = np.max(
            np.array(
                [
                    [len(modes) for modes in all_modes]
                    for subplot_index in range(num_cols)
                    for mode_time_series in all_mode_time_series[
                        subplot_index
                    ].values()
                    for _, all_modes in mode_time_series
                ]
            ),
            axis=0,
        )
        mode_cmap = [
            LinearSegmentedColormap.from_list(
                "Modes",
                ["black", props_dim[d].get("color", "black")],
                N=max_num_modes[d],
            )
            for d in range(domain.dim)
        ]
        for subplot_index in range(num_cols):
            for element_id, mode_time_series in all_mode_time_series[
                subplot_index
            ].items():
                times = np.array([time for time, _ in mode_time_series])
                for d in range(domain.dim):
                    ax = axes[d][subplot_index]
                    for mode in range(max_num_modes[d]):
                        mode_time_series_i = np.array(
                            [
                                (
                                    all_modes[d][mode]
                                    if len(all_modes[d]) > mode
                                    else np.nan
                                )
                                for _, all_modes in mode_time_series
                            ]
                        )
                        color = mode_cmap[d](mode / (max_num_modes[d] - 1))
                        ax.semilogy(
                            times,
                            mode_time_series_i,
                            color=color,
                            zorder=30 + d,
                        )
        # Plot colorbars as legend
        import matplotlib.cm
        import matplotlib.colors

        for d in range(domain.dim):
            colorbar = plt.colorbar(
                matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(0, max_num_modes[d]),
                    cmap=mode_cmap[d],
                ),
                ax=axes[d],
                ticks=list(range(max_num_modes[d])),
                label=dimension_labels[d] + " Mode",
            )
            colorbar.ax.invert_yaxis()
    else:
        # Annotate the max truncation error. Also serves as a legend.
        for subplot_index, ax in enumerate(axes[0]):
            for d in range(domain.dim):
                ax.axhline(
                    max_error[subplot_index][d], **props_dim[d], zorder=20 + d
                )
                ax.annotate(
                    dimension_labels[d],
                    xy=(0, max_error[subplot_index][d]),
                    xytext=((2 * d + 0.5) * plt.rcParams["font.size"], 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    bbox=dict(
                        fc="white",
                        ec=props_dim[d].get("color", "black"),
                        pad=2.0,
                    ),
                    zorder=40 + d,
                )

    # Set plot titles
    for subplot_index, ax in enumerate(axes[0]):
        ax.set_title(block_or_group_names[subplot_index], loc="left")
        num_elements_i = (
            len(all_mode_time_series[subplot_index])
            if plot_over_time
            else num_elements[subplot_index]
        )
        ax.set_title(
            f"{num_elements_i} element" + "s"[: num_elements_i != 1],
            loc="right",
        )

    for axes_row in axes:
        for ax in axes_row:
            # Draw grid lines
            ax.grid(which="both", zorder=0)
            # Allow only integer ticks for modes
            if not plot_over_time:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add y-labels to the leftmost subplots
    if plot_over_time:
        for d, ax in enumerate(axes):
            ax[0].set_ylabel(
                r"Power monitors $P_{q_" + dimension_labels[d].strip("$") + "}$"
            )
    else:
        axes[0][0].set_ylabel(r"Power monitors $P_{q_{\hat{\imath}}}$")

    # Add x-label spanning all subplots
    ax_colspan = fig.add_subplot(111, frameon=False)
    ax_colspan.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_colspan.grid(False)
    ax_colspan.set_xlabel("Time" if plot_over_time else "Mode number")


@click.command(name="power-monitors")
@open_volfiles_command(obs_id_required=False, multiple_vars=True)
@click.option(
    "--list-blocks",
    is_flag=True,
    help="Print available blocks and block groups and exit.",
)
@click.option(
    "--block",
    "-b",
    "block_or_group_names",
    multiple=True,
    help=(
        "Name of block or block group to analyze. "
        "Can be specified multiple times to plot several block(groups) at once."
    ),
)
@click.option(
    "--elements",
    "-e",
    "element_patterns",
    multiple=True,
    help=(
        "Include only elements that match the specified glob "
        "pattern, like 'B*,(L1I*,L0I0,L0I0)'. "
        "Can be specified multiple times, in which case elements "
        "are included that match _any_ of the specified "
        "patterns. If unspecified, include all elements in the blocks."
    ),
)
@click.option(
    "--list-elements",
    is_flag=True,
    help=(
        "List all elements in the specified blocks subject to "
        "'--elements' / '-e' patterns."
    ),
)
@click.option(
    "--over-time", "-T", is_flag=True, help="Plot power monitors over time."
)
@click.option(
    "--skip-filtered-modes",
    type=int,
    default=0,
    help=(
        "Skip this number of highest modes. Useful if the highest modes are"
        " filtered, zeroing them out."
    ),
)
# Plotting options
@click.option("--figsize", nargs=2, type=float, help="Figure size in inches.")
@apply_stylesheet_command()
@show_or_save_plot_command()
def plot_power_monitors_command(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    vars,
    list_blocks,
    block_or_group_names,
    list_elements,
    element_patterns,
    over_time,
    **kwargs,
):
    """Plot power monitors from volume data

    Reads volume data in the 'H5_FILES' and computes power monitors, which are
    essentially the spectral modes in each dimension of the grid. They give an
    indication how well the spectral expansion resolves fields on the grid.
    Power monitors are computed for all tensor components selected with the
    '--var' / '-y' option, and combined as an L2 norm.

    One subplot is created for every selected '--block' / '-b'. This can be a
    single block name, or a block group defined by the domain (such as all six
    wedges in a spherical shell). The power monitors in every logical direction
    of the grid are plotted for all elements in the block or block group. The
    logical directions are labeled "xi", "eta" and "zeta", and their orientation
    is defined by the coordinate maps in the domain. For example, see the
    documentation of the 'Wedge' map to understand which logical direction is
    radial in spherical shells.
    """
    if over_time == (obs_id is not None):
        raise click.UsageError(
            "Specify an observation '--step' or '--time', or specify"
            " '--over-time' (but not both)."
        )

    # Print available blocks and groups
    open_h5_file = spectre_h5.H5File(h5_files[0], "r")
    volfile = open_h5_file.get_vol(subfile_name)
    dim = volfile.get_dimension()
    any_obs_id = (
        obs_id if obs_id is not None else volfile.list_observation_ids()[0]
    )
    domain = deserialize_domain[dim](volfile.get_domain(any_obs_id))
    all_block_groups = list(domain.block_groups.keys())
    all_block_names = [block.name for block in domain.blocks]
    if list_blocks:
        import rich.columns

        rich.print(rich.columns.Columns(all_block_groups + all_block_names))
        return
    elif not block_or_group_names:
        raise RequiredChoiceError(
            (
                "Specify '--block' / '-b' to select (possibly multiple) blocks"
                " or block groups to analyze."
            ),
            choices=all_block_groups + all_block_names,
        )
    # Validate block and group names
    for name in block_or_group_names:
        if not (name in all_block_groups or name in all_block_names):
            raise RequiredChoiceError(
                f"'{name}' matches no block or block group.",
                choices=all_block_groups + all_block_names,
            )

    # Print available elements IDs
    if not element_patterns:
        # Don't apply any filters when no element patterns were specified
        element_patterns = None
    if list_elements:
        all_element_ids = sorted(
            set(
                element.id
                for element in iter_elements(
                    open_volfiles(h5_files, subfile_name, obs_id),
                    obs_id,
                    element_patterns=element_patterns,
                )
            )
        )
        # Print grouped by block
        import rich.console

        console = rich.console.Console()
        for i, block_name in enumerate(block_or_group_names):
            element_ids = [
                stripped_element_name(element_id)
                for element_id in all_element_ids
                if find_block_or_group(
                    element_id.block_id, block_or_group_names, domain
                )
                == i
            ]
            console.rule(
                f"[bold]{block_name}[/bold] ({len(element_ids)} elements)"
            )
            console.print(rich.columns.Columns(element_ids))
        return

    # Close the H5 file because we're done with preprocessing
    open_h5_file.close()

    # Plot!
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(h5_files) == 1),
    )
    task_id = progress.add_task("Processing files", total=len(h5_files))
    volfiles_progress = progress.track(
        open_volfiles(h5_files, subfile_name, obs_id), task_id=task_id
    )
    with progress:
        plot_power_monitors(
            volfiles_progress,
            obs_id=obs_id,
            tensor_components=vars,
            domain=domain,
            block_or_group_names=block_or_group_names,
            element_patterns=element_patterns,
            **kwargs,
        )
        progress.update(task_id, completed=len(h5_files))


if __name__ == "__main__":
    plot_power_monitors_command(help_option_names=["-h", "--help"])
