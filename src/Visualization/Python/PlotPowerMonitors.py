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
from spectre.DataStructures.Tensor import Frame, Scalar, tnsr
from spectre.Domain import Domain, deserialize_domain
from spectre.IO.H5 import open_volfiles, open_volfiles_command, parse_point
from spectre.IO.H5.IterElements import iter_elements, stripped_element_name
from spectre.IO.H5.TransformVolumeData import get_tensor_component_names
from spectre.NumericalAlgorithms.LinearOperators import power_monitors
from spectre.Spectral import Basis
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.Visualization.Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)

logger = logging.getLogger(__name__)


def gh_sh_tensor_component_names(
    spacetime_metric_name: str, pi_name: str, phi_name: str
):
    return (
        get_tensor_component_names(
            spacetime_metric_name, tnsr.aa[DataVector, 3, Frame.Inertial]
        )
        + get_tensor_component_names(
            pi_name, tnsr.aa[DataVector, 3, Frame.Inertial]
        )
        + get_tensor_component_names(
            phi_name, tnsr.iaa[DataVector, 3, Frame.Inertial]
        )
    )


def sw_sh_tensor_component_names(psi_name: str, pi_name: str, phi_name: str):
    """Tensor components of the (Curved)ScalarWave evolved variables.

    CurvedScalarWave evolves the same variables as ScalarWave, so this is
    shared between the 'sw' and 'csw' systems. See `plot_sw_power_monitors`.
    """
    return (
        get_tensor_component_names(psi_name, Scalar[DataVector])
        + get_tensor_component_names(pi_name, Scalar[DataVector])
        + get_tensor_component_names(
            phi_name, tnsr.i[DataVector, 3, Frame.Inertial]
        )
    )


def _plot_sh_power_monitors_impl(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: int,
    block_or_group_names: Sequence[str],
    domain: Domain[3],
    tensor_components: Sequence[str],
    get_monitors,
    variables_to_plot: Sequence[str],
    element_patterns: Optional[Sequence[str]] = None,
    fixed_y_limits: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Power monitors at observation {obs_id}",
):
    num_cols = len(variables_to_plot)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_cols,
        figsize=figsize or (num_cols * 4, 7),
        sharey="row",
        squeeze=False,
    )
    plotted_elements = 0
    for element, tensor_data in iter_elements(
        volfiles, obs_id, tensor_components, element_patterns=element_patterns
    ):
        if any(
            basis == Basis.FiniteDifference for basis in element.mesh.basis()
        ):
            continue
        if (
            find_block_or_group(
                element.id.block_id, block_or_group_names, domain
            )
            is None
        ):
            continue
        if tensor_data.dtype != np.float64:
            tensor_data = tensor_data.astype(np.float64)
        monitors = get_monitors(
            tensor_data,
            element.mesh,
            element.id,
            domain,
            element.time,
            element.functions_of_time or {},
        )
        if monitors is None:
            continue
        label = stripped_element_name(element.id)
        for col, variable_name in enumerate(variables_to_plot):
            radial = np.asarray(monitors[variable_name]["radial"])
            angular = np.asarray(monitors[variable_name]["angular"])
            axes[0][col].semilogy(radial, label=label)
            axes[1][col].semilogy(angular, label=label)
        plotted_elements += 1

    for col, variable_name in enumerate(variables_to_plot):
        axes[0][col].set_title(variable_name)
        axes[1][col].set_xlabel("Mode number")
    axes[0][0].set_ylabel("Radial power")
    axes[1][0].set_ylabel("Angular power")
    for ax in axes.ravel():
        ax.grid(which="both")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if fixed_y_limits is not None:
            ax.set_ylim(*fixed_y_limits)
    if 0 < plotted_elements <= 12:
        axes[0][-1].legend(loc="best", fontsize="small")
    fig.suptitle(title.format(obs_id=obs_id))
    fig.tight_layout()
    return fig


def plot_gh_power_monitors(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: int,
    block_or_group_names: Sequence[str],
    domain: Domain[3],
    spacetime_metric_name: str = "SpacetimeMetric",
    pi_name: str = "Pi",
    phi_name: str = "Phi",
    element_patterns: Optional[Sequence[str]] = None,
    variables_to_plot: Sequence[str] = ("SpacetimeMetric", "Pi", "Phi"),
    fixed_y_limits: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    if domain.dim != 3:
        raise click.UsageError("GH power monitors require 3D volume data.")
    from spectre.Evolution.Systems.GeneralizedHarmonic import (
        gh_b3_power_monitors,
        gh_shell_power_monitors,
    )

    tensor_components = gh_sh_tensor_component_names(
        spacetime_metric_name, pi_name, phi_name
    )
    metric_type = tnsr.aa[DataVector, 3, Frame.Inertial]
    pi_type = tnsr.aa[DataVector, 3, Frame.Inertial]
    phi_type = tnsr.iaa[DataVector, 3, Frame.Inertial]
    metric_size = metric_type.size
    pi_size = pi_type.size

    def get_monitors(tensor_data, mesh, element_id, domain, time, fot):
        if all(basis == Basis.ZernikeB3 for basis in mesh.basis()):
            monitor_fn = gh_b3_power_monitors
        elif any(basis == Basis.SphericalHarmonic for basis in mesh.basis()):
            monitor_fn = gh_shell_power_monitors
        else:
            return None
        spacetime_metric = metric_type(tensor_data[:metric_size])
        pi = pi_type(tensor_data[metric_size : metric_size + pi_size])
        phi = phi_type(tensor_data[metric_size + pi_size :])
        return monitor_fn(
            spacetime_metric, pi, phi, mesh, element_id, domain, time, fot
        )

    return _plot_sh_power_monitors_impl(
        volfiles,
        obs_id,
        block_or_group_names,
        domain,
        tensor_components,
        get_monitors,
        variables_to_plot,
        element_patterns=element_patterns,
        fixed_y_limits=fixed_y_limits,
        figsize=figsize,
        title="GH power monitors at observation {obs_id}",
    )


def plot_sw_power_monitors(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: int,
    block_or_group_names: Sequence[str],
    domain: Domain[3],
    psi_name: str = "Psi",
    pi_name: str = "Pi",
    phi_name: str = "Phi",
    element_patterns: Optional[Sequence[str]] = None,
    variables_to_plot: Sequence[str] = ("Psi", "Pi", "Phi"),
    fixed_y_limits: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_prefix: str = "SW",
):
    """Plot spherical-harmonic power monitors for a scalar wave system.

    This serves both the ScalarWave ('sw') and CurvedScalarWave ('csw')
    systems. CurvedScalarWave evolves the same variables as ScalarWave --
    `Psi` and `Pi` are `Scalar<DataVector>` and `Phi` is
    `tnsr::i<DataVector, 3>` in both. The power monitors depend only on
    the numerical data and the mesh, so there is no separate `CurvedScalarWave`
    implementation.
    """
    if domain.dim != 3:
        raise click.UsageError("SW power monitors require 3D volume data.")
    from spectre.Evolution.Systems.ScalarWave import (
        sw_b3_power_monitors,
        sw_shell_power_monitors,
    )

    tensor_components = sw_sh_tensor_component_names(
        psi_name, pi_name, phi_name
    )
    psi_type = Scalar[DataVector]
    pi_type = Scalar[DataVector]
    phi_type = tnsr.i[DataVector, 3, Frame.Inertial]
    psi_size = psi_type.size
    pi_size = pi_type.size

    def get_monitors(tensor_data, mesh, element_id, domain, time, fot):
        if all(basis == Basis.ZernikeB3 for basis in mesh.basis()):
            monitor_fn = sw_b3_power_monitors
        elif any(basis == Basis.SphericalHarmonic for basis in mesh.basis()):
            monitor_fn = sw_shell_power_monitors
        else:
            return None
        psi = psi_type(tensor_data[:psi_size])
        pi = pi_type(tensor_data[psi_size : psi_size + pi_size])
        phi = phi_type(tensor_data[psi_size + pi_size :])
        return monitor_fn(psi, pi, phi, mesh, element_id, domain, time, fot)

    return _plot_sh_power_monitors_impl(
        volfiles,
        obs_id,
        block_or_group_names,
        domain,
        tensor_components,
        get_monitors,
        variables_to_plot,
        element_patterns=element_patterns,
        fixed_y_limits=fixed_y_limits,
        figsize=figsize,
        title=title_prefix + " power monitors at observation {obs_id}",
    )


# 'csw' (CurvedScalarWave) deliberately reuses 'plot_sw_power_monitors'
_SH_SYSTEMS = {
    "gh": {
        "variables": ("SpacetimeMetric", "Pi", "Phi"),
        "plot_fn": "plot_gh_power_monitors",
    },
    "sw": {
        "variables": ("Psi", "Pi", "Phi"),
        "plot_fn": "plot_sw_power_monitors",
    },
    "csw": {
        "variables": ("Psi", "Pi", "Phi"),
        "plot_fn": "plot_sw_power_monitors",
    },
}


def _detect_sh_system(open_h5_file, volfile) -> str:
    """Detect the evolution system from the H5 file.

    Tries the 'Executable' field in the embedded input YAML first, then falls
    back to inspecting the tensor component names in the volume data.

    Note that ScalarWave and CurvedScalarWave volume data are
    indistinguishable by tensor component names, so only the 'Executable' field
    can tell them apart. The component fallback gives 'sw', which computes the
    same operations as `csw`. Use '--sh-system csw' to label the plot
    explicitly.

    Returns one of the keys of _SH_SYSTEMS.
    """
    import yaml

    input_src = open_h5_file.input_source()
    if input_src:
        metadata = next(yaml.safe_load_all(input_src), None)
        if metadata:
            exe = metadata.get("Executable", "")
            if "CurvedScalarWave" in exe:
                return "csw"
            if exe.startswith("EvolveScalarWave"):
                return "sw"
            if exe.startswith("EvolveGh") and not exe.startswith(
                "EvolveGhValenciaDivClean"
            ):
                return "gh"

    # Fall back to component inspection
    obs_ids = volfile.list_observation_ids()
    if obs_ids:
        components = set(volfile.list_tensor_components(obs_ids[0]))
        if "Psi" in components:
            return "sw"
        if any(c.startswith("SpacetimeMetric") for c in components):
            return "gh"

    raise click.UsageError(
        "Cannot auto-detect the evolution system from the H5 file. "
        "Specify '--sh-system {gh,sw}'."
    )


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

    shown_dtype_warning_once = False
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
        if tensor_data.dtype != np.float64:
            if not shown_dtype_warning_once:
                logger.warning(
                    "Tensor data is not double precision. Power monitors"
                    " will be inaccurate below the precision of the data."
                )
                shown_dtype_warning_once = True
            tensor_data = tensor_data.astype(np.float64)
        all_modes = None
        for component in tensor_data:
            modes = power_monitors(DataVector(component), element.mesh)
            if all_modes is None:
                all_modes = [
                    np.zeros(len(modes_dim) - skip_filtered_modes)
                    for modes_dim in modes
                ]
            for d, modes_dim in enumerate(modes):
                num_modes = len(modes_dim) - skip_filtered_modes
                all_modes[d] += np.array(modes_dim)[:num_modes] ** 2
        if all_modes is None:
            continue
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
            f"{num_elements_i} element" + ("" if num_elements_i == 1 else "s"),
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
@open_volfiles_command(
    obs_id_required=False, vars_required=False, multiple_vars=True
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
    "--sh",
    "sh",
    is_flag=True,
    help=(
        "Compute TensorYlm-basis power monitors using the evolution system "
        "detected from the H5 file (GH: SpacetimeMetric/Pi/Phi; SW and CSW: "
        "Psi/Pi/Phi). Handles shell (SphericalHarmonic) and filled-sphere "
        "(ZernikeB3) elements automatically."
    ),
)
@click.option(
    "--sh-system",
    type=click.Choice(list(_SH_SYSTEMS.keys())),
    default=None,
    help=(
        "Override auto-detection of the evolution system for '--sh'. "
        "Auto-detected from the 'Executable' field in the H5 input source "
        "or from the tensor component names present in the data."
    ),
)
@click.option(
    "--sh-variable",
    "sh_variables",
    multiple=True,
    type=click.Choice(
        sorted({v for s in _SH_SYSTEMS.values() for v in s["variables"]})
    ),
    help=(
        "Variable to plot in '--sh' mode. Can be specified multiple times. "
        "Defaults to all variables for the detected system."
    ),
)
@click.option(
    "--sh-frame-prefix",
    type=click.Path(file_okay=False, dir_okay=False, writable=True),
    help=(
        "When used with '--sh --over-time', write one PNG frame per "
        "observation with this filename prefix."
    ),
)
@click.option(
    "--sh-spacetime-metric",
    default="SpacetimeMetric",
    show_default=True,
    help="Volume-data tensor name for the GH SpacetimeMetric.",
)
@click.option(
    "--sh-psi",
    default="Psi",
    show_default=True,
    help="Volume-data tensor name for the SW/CSW Psi.",
)
@click.option(
    "--sh-pi",
    default="Pi",
    show_default=True,
    help="Volume-data tensor name for Pi.",
)
@click.option(
    "--sh-phi",
    default="Phi",
    show_default=True,
    help="Volume-data tensor name for Phi.",
)
@click.option(
    "--fixed-y-limits",
    nargs=2,
    type=float,
    help="Fixed y-axis limits for GH spherical harmonic movie frames.",
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
    sh,
    sh_system,
    sh_variables,
    sh_frame_prefix,
    sh_spacetime_metric,
    sh_psi,
    sh_pi,
    sh_phi,
    fixed_y_limits,
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
    if not sh and not vars:
        raise click.UsageError(
            "Specify '--var' / '-y' to select a variable to plot, or use "
            "'--sh'."
        )
    if sh and over_time and not sh_frame_prefix:
        raise click.UsageError(
            "Specify '--sh-frame-prefix' when using "
            "'--sh --over-time' so one plot can be written for each "
            "observation."
        )

    # Print available blocks and groups
    open_h5_file = spectre_h5.H5File(h5_files[0], "r")
    volfile = open_h5_file.get_vol(subfile_name)
    dim = volfile.get_dimension()
    domain = deserialize_domain[dim](volfile.get_domain())
    all_block_groups = list(domain.block_groups.keys())
    all_block_names = [block.name for block in domain.blocks]
    if list_blocks:
        import rich.columns

        rich.print(rich.columns.Columns(all_block_groups + all_block_names))
        open_h5_file.close()
        return
    elif not block_or_group_names:
        raise RequiredChoiceError(
            "Specify '--block' / '-b' to select (possibly multiple) blocks"
            " or block groups to analyze.",
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
        open_h5_file.close()
        return

    if sh:
        kwargs.pop("skip_filtered_modes")
        detected_system = sh_system or _detect_sh_system(open_h5_file, volfile)
        system_vars = _SH_SYSTEMS[detected_system]["variables"]
        # Validate user-requested variables against the detected system
        for v in sh_variables:
            if v not in system_vars:
                raise click.UsageError(
                    f"Variable '{v}' is not available for the detected system"
                    f" '{detected_system}'. Available: {list(system_vars)}."
                )
        variables_to_plot = sh_variables or system_vars

        # Build system-specific keyword args for the plot function
        if detected_system == "gh":
            plot_fn = plot_gh_power_monitors
            plot_kwargs = dict(
                spacetime_metric_name=sh_spacetime_metric,
                pi_name=sh_pi,
                phi_name=sh_phi,
            )
        else:
            assert detected_system in ("sw", "csw")
            # CurvedScalarWave evolves the same variables as ScalarWave, so
            # the ScalarWave monitors work for both
            plot_fn = plot_sw_power_monitors
            plot_kwargs = dict(
                psi_name=sh_psi,
                pi_name=sh_pi,
                phi_name=sh_phi,
                title_prefix=detected_system.upper(),
            )

        if over_time:
            all_obs_ids = volfile.list_observation_ids()
            all_obs_times = [
                volfile.get_observation_value(obs_id_i)
                for obs_id_i in all_obs_ids
            ]
            open_h5_file.close()
            import rich.progress

            progress = rich.progress.Progress(
                rich.progress.TextColumn(
                    "[progress.description]{task.description}"
                ),
                rich.progress.BarColumn(),
                rich.progress.MofNCompleteColumn(),
                rich.progress.TimeRemainingColumn(),
            )
            task = progress.add_task(
                f"Writing {detected_system.upper()} frames",
                total=len(all_obs_ids),
            )
            with progress:
                for obs_id_i, obs_time_i in zip(all_obs_ids, all_obs_times):
                    fig = plot_fn(
                        open_volfiles(h5_files, subfile_name, obs_id_i),
                        obs_id=obs_id_i,
                        domain=domain,
                        block_or_group_names=block_or_group_names,
                        element_patterns=element_patterns,
                        variables_to_plot=variables_to_plot,
                        fixed_y_limits=fixed_y_limits,
                        **plot_kwargs,
                        **kwargs,
                    )
                    fig.savefig(
                        f"{sh_frame_prefix}_"
                        f"{obs_id_i:012d}_t{obs_time_i:.12g}.png"
                    )
                    plt.close(fig)
                    progress.update(task, advance=1)
            return

        open_h5_file.close()
        return plot_fn(
            open_volfiles(h5_files, subfile_name, obs_id),
            obs_id=obs_id,
            domain=domain,
            block_or_group_names=block_or_group_names,
            element_patterns=element_patterns,
            variables_to_plot=variables_to_plot,
            fixed_y_limits=fixed_y_limits,
            **plot_kwargs,
            **kwargs,
        )

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
