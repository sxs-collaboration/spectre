# Distributed under the MIT License.
# See LICENSE.txt for details.

import fnmatch
import logging
import os
from itertools import cycle
from typing import Iterable, Optional, Sequence, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import Domain, deserialize_domain
from spectre.IO.H5.IterElements import iter_elements, stripped_element_name
from spectre.NumericalAlgorithms.LinearOperators import power_monitors
from spectre.Spectral import Basis

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
):
    plot_over_time = obs_id is None
    # One column per block or group
    num_cols = len(block_or_group_names)
    # One row per dimension if plotted over time to declutter the plots
    num_rows = domain.dim if plot_over_time else 1
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * 4, num_rows * 4),
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
            np.zeros(element.mesh.extents(d)) for d in range(element.dim)
        ]
        for component in tensor_data:
            modes = power_monitors(DataVector(component), element.mesh)
            for d, modes_dim in enumerate(modes):
                all_modes[d] += modes_dim**2
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

    # Draw grid lines
    for axes_row in axes:
        for ax in axes_row:
            ax.grid(which="both", zorder=0)

    # Add x-label spanning all subplots
    ax_colspan = fig.add_subplot(111, frameon=False)
    ax_colspan.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_colspan.grid(False)
    ax_colspan.set_xlabel("Time" if plot_over_time else "Mode number")


def parse_step(ctx, param, value):
    if value is None:
        return None
    if value.lower() == "last":
        return -1
    return int(value)


@click.command()
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help="Name of volume data subfile within the h5 files.",
)
@click.option(
    "--step",
    callback=parse_step,
    help=(
        "Observation step number. Specify '-1' or 'last' "
        "for the last step in the file (default). "
        "Mutually exclusive with '--time'."
    ),
)
@click.option(
    "--time",
    type=float,
    help=(
        "Observation time. "
        "The observation step closest to the specified "
        "time is selected. "
        "Mutually exclusive with '--step'."
    ),
)
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
        "Names of blocks or block groups to analyze. "
        "Can be specified multiple times."
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
        "patterns."
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
    "--list-vars",
    "-l",
    is_flag=True,
    help="Print available variables and exit.",
)
@click.option(
    "--var",
    "-y",
    "vars_patterns",
    multiple=True,
    help=(
        "Variables to plot. List any tensor components "
        "in the volume data file, such as 'Shift_x'. "
        "Also accepts glob patterns like 'Shift_*'."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help=(
        "Name of the output plot file. If unspecified, the plot is "
        "shown interactively, which only works on machines with a "
        "window server."
    ),
)
@click.option(
    "--stylesheet",
    "-s",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    envvar="SPECTRE_MPL_STYLESHEET",
    help=(
        "Select a matplotlib stylesheet for customization of the plot, such "
        "as linestyle cycles, linewidth, fontsize, legend, etc. "
        "The stylesheet can also be set with the 'SPECTRE_MPL_STYLESHEET' "
        "environment variable."
    ),
)
def plot_power_monitors_command(
    h5_files,
    subfile_name,
    step,
    time,
    list_blocks,
    block_or_group_names,
    list_elements,
    element_patterns,
    list_vars,
    vars_patterns,
    output,
    stylesheet,
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
    # Script should be a noop if input files are empty
    if not h5_files:
        return

    open_h5_files = [spectre_h5.H5File(filename, "r") for filename in h5_files]

    # Print available subfile names and exit
    if not subfile_name:
        import rich.columns

        rich.print(rich.columns.Columns(open_h5_files[0].all_vol_files()))
        return

    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name.rstrip(".vol")
    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name

    volfiles = [h5file.get_vol(subfile_name) for h5file in open_h5_files]
    obs_ids = volfiles[0].list_observation_ids()
    obs_values = list(map(volfiles[0].get_observation_value, obs_ids))
    dim = volfiles[0].get_dimension()

    # Select observation
    if step is None and time is None:
        # Plot power monitors over time
        obs_id = None
    elif step is not None and time is not None:
        raise click.UsageError(
            f"Specify either '--step' (in [0, {len(obs_ids) - 1}], or -1) or "
            f"'--time' (in [{obs_values[0]:g}, {obs_values[-1]:g}]), "
            "or neither."
        )
    elif step is None:
        # Find closest observation to specified time
        step = np.argmin(np.abs(time - np.array(obs_values)))
        obs_value = obs_values[step]
        if obs_value != time:
            logger.info(
                f"Selected closest observation to t = {time}: "
                f"step {step} at t = {obs_value:g}"
            )
        obs_id = obs_ids[step]
    else:
        # Select the specified observation
        obs_id = obs_ids[step]

    # Print available blocks and groups
    any_obs_id = obs_id if obs_id is not None else obs_ids[0]
    domain = deserialize_domain[dim](volfiles[0].get_domain(any_obs_id))
    all_block_groups = list(domain.block_groups.keys())
    all_block_names = [block.name for block in domain.blocks]
    if list_blocks or not block_or_group_names:
        import rich.columns

        rich.print(rich.columns.Columns(all_block_groups + all_block_names))
        return
    # Validate block and group names
    for name in block_or_group_names:
        if not (name in all_block_groups or name in all_block_names):
            raise click.UsageError(
                f"'{name}' matches no block or group name. "
                f"Available names are: {all_block_groups + all_block_names}"
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
                    volfiles, obs_id, element_patterns=element_patterns
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

    # Print available variables and exit
    all_vars = volfiles[0].list_tensor_components(any_obs_id)
    if list_vars or not vars_patterns:
        import rich.columns

        rich.print(rich.columns.Columns(all_vars))
        return
    # Expand globs in vars
    vars = []
    for var_pattern in vars_patterns:
        matched_vars = fnmatch.filter(all_vars, var_pattern)
        if not matched_vars:
            raise click.UsageError(
                f"The pattern '{var_pattern}' matches no variables. "
                f"Available variables are: {all_vars}"
            )
        vars.extend(matched_vars)
    # Remove duplicates. Ordering is lost, but that's not important here.
    vars = list(set(vars))

    # Apply stylesheets
    stylesheets = [os.path.join(os.path.dirname(__file__), "plots.mplstyle")]
    if stylesheet is not None:
        stylesheets.append(stylesheet)
    plt.style.use(stylesheets)

    # Plot!
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(volfiles) == 1),
    )
    task_id = progress.add_task("Processing files")
    volfiles_progress = progress.track(volfiles, task_id=task_id)
    with progress:
        plot_power_monitors(
            volfiles_progress,
            obs_id=obs_id,
            tensor_components=vars,
            domain=domain,
            block_or_group_names=block_or_group_names,
            element_patterns=element_patterns,
        )
        progress.update(task_id, completed=len(volfiles))

    if output:
        plt.savefig(output)
    else:
        if not os.environ.get("DISPLAY"):
            logger.warning(
                "No 'DISPLAY' environment variable is configured so plotting "
                "interactively is unlikely to work. Write the plot to a file "
                "with the --output/-o option."
            )
        plt.show()


if __name__ == "__main__":
    plot_power_monitors_command(help_option_names=["-h", "--help"])
