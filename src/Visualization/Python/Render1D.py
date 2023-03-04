#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from itertools import cycle
from typing import Dict, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import spectre.IO.H5 as spectre_h5
from scipy.interpolate import lagrange
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr
from spectre.Domain import (FunctionOfTime, deserialize_domain,
                            deserialize_functions_of_time)
from spectre.IO.H5.IterElements import Element, iter_elements
from spectre.Spectral import Basis
from spectre.Visualization.PlotDatFile import parse_functions

logger = logging.getLogger(__name__)


def get_bounds(volfiles, obs_ids, vars):
    """Get the bounds in both x and y over all observations, vars, and files"""
    x_bounds = [np.inf, -np.inf]
    y_bounds = [np.inf, -np.inf]
    for obs_id in obs_ids:
        for element, vars_data in iter_elements(volfiles, obs_id, vars):
            x_bounds[0] = min(x_bounds[0],
                              np.min(element.inertial_coordinates))
            x_bounds[1] = max(x_bounds[1],
                              np.max(element.inertial_coordinates))
            y_bounds[0] = min(y_bounds[0], np.nanmin(vars_data))
            y_bounds[1] = max(y_bounds[1], np.nanmax(vars_data))
    return x_bounds, y_bounds


def plot_element(element: Element,
                 vars_data: np.ndarray,
                 vars: Dict[str, str],
                 var_props: dict,
                 show_collocation_points: bool,
                 show_element_boundaries: bool,
                 show_basis_polynomials: bool,
                 time: Optional[float] = None,
                 functions_of_time: Optional[Dict[str, FunctionOfTime]] = None,
                 logical_space: np.ndarray = np.linspace(-1, 1, 50),
                 handles: Optional[dict] = None):
    """Plot a 1D element, or update the plot in an animation"""

    # We store plots in these dicts so we can update them later in an animation
    # by calling `plot_element` again
    if handles is not None:
        element_handles = handles.setdefault(element.id, dict())
        point_handles = element_handles.setdefault("point", dict())
        boundary_handles = element_handles.setdefault("boundary", dict())
        var_handles = element_handles.setdefault("var", dict())
        interpolation_handles = element_handles.setdefault("interp", dict())
        basis_handles = element_handles.setdefault("basis", dict())

    # We collect legend items and return them from this function
    legend_items = dict()

    inertial_coords = np.array(element.inertial_coordinates)[0]

    # Plot collocation points
    if show_collocation_points:
        for i, coord in enumerate(inertial_coords):
            if handles and i in point_handles:
                point_handles[i].set_xdata(coord)
            else:
                point_handles[i] = plt.axvline(coord,
                                               color="black",
                                               ls="dotted",
                                               alpha=0.2)
        legend_items["Collocation points"] = point_handles[0]
        # Clean up leftover handles
        if handles:
            all_point_handles = sorted(list(point_handles.keys()))
            for i in all_point_handles[all_point_handles.index(i) + 1:]:
                point_handles.pop(i).remove()

    # Plot element boundaries
    if show_element_boundaries:
        logical_boundaries = tnsr.I[DataVector, 1,
                                    Frame.ElementLogical](np.array([[-1.,
                                                                     1.]]))
        element_boundaries = np.asarray(
            element.map(logical_boundaries,
                        time=time,
                        functions_of_time=functions_of_time))[0]
        for i, coord in enumerate(element_boundaries):
            if handles and i in boundary_handles:
                boundary_handles[i].set_xdata(coord)
            else:
                boundary_handles[i] = plt.axvline(coord, color="black")
        legend_items["Element boundaries"] = boundary_handles[0]

    # Prepare Lagrange interpolation
    # Only show Lagrange interpolation for spectral elements. Finite-difference
    # elements just show the data points.
    show_lagrange_interpolation = all([
        basis == Basis.Legendre or basis == Basis.Chebyshev
        for basis in element.mesh.basis()
    ])
    if show_lagrange_interpolation:
        # These are the control points for the Lagrange interpolation
        logical_coords = np.array(element.logical_coordinates)[0]
        # These are the points where we plot the interpolation
        logical_space_tensor = tnsr.I[DataVector, 1, Frame.ElementLogical](
            np.expand_dims(logical_space, axis=0))
        inertial_space = np.asarray(
            element.map(logical_space_tensor,
                        time=time,
                        functions_of_time=functions_of_time))[0]

    # Plot selected variables
    for (var, label), var_data in zip(vars.items(), vars_data):
        # Plot data points
        if handles and var in var_handles:
            var_handles[var].set_data(inertial_coords, var_data)
        else:
            var_handles[var] = plt.plot(inertial_coords,
                                        var_data,
                                        marker=".",
                                        ls="none",
                                        **var_props[var])[0]
        legend_items[label] = var_handles[var]
        # Plot Lagrange interpolation
        if show_lagrange_interpolation:
            interpolant = lagrange(logical_coords, var_data)
            if handles and var in interpolation_handles:
                interpolation_handles[var].set_data(inertial_space,
                                                    interpolant(logical_space))
            else:
                interpolation_handles[var] = plt.plot(
                    inertial_space, interpolant(logical_space),
                    **var_props[var])[0]
            # Plot polynomial basis
            if show_basis_polynomials:
                unit_weights = np.eye(len(logical_coords))
                for xi_i in range(len(logical_coords)):
                    basis_id = (var, xi_i)
                    basis_polynomial = lagrange(
                        logical_coords, var_data[xi_i] * unit_weights[xi_i])
                    if handles and basis_id in basis_handles:
                        basis_handles[basis_id].set_data(
                            inertial_space, basis_polynomial(logical_space))
                    else:
                        basis_handles[basis_id] = plt.plot(
                            inertial_space,
                            basis_polynomial(logical_space),
                            color="black",
                            alpha=0.2)[0]
        elif handles:
            # Clean up leftover handles
            if var in interpolation_handles:
                interpolation_handles.pop(var).remove()
            for basis_id in list(basis_handles.keys()):
                basis_handles.pop(basis_id).remove()
    if show_lagrange_interpolation and show_basis_polynomials:
        legend_items["Lagrange basis"] = basis_handles[basis_id]

    return legend_items


@click.command()
@click.argument("h5_files",
                nargs=-1,
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True))
@click.option("--subfile-name",
              "-d",
              help=("Name of subfile within h5 file containing "
                    "1D volume data to be rendered."))
@click.option(
    "--var",
    "-y",
    "vars",
    multiple=True,
    callback=parse_functions,
    help=("Name of variable to plot, e.g. 'Psi' or 'Error(Psi)'. "
          "Can be specified multiple times. "
          "If unspecified, plot all available variables. "
          "Labels for variables can be specified as key-value pairs such as "
          "'Error(Psi)=$L_2(\\psi)$'. Remember to wrap the key-value pair in "
          "quotes on the command line to avoid issues with special characters "
          "or spaces."))
@click.option("--list-vars",
              "-l",
              is_flag=True,
              help="Print available variables and exit.")
@click.option("--step",
              type=int,
              help=("If specified, renders the integer observation step "
                    "instead of an animation. Set to '-1' for the last step."))
@click.option("-o",
              "--output",
              help=("Set the name of the output file you want "
                    "written. For animations this saves an mp4 file and "
                    "for stills a pdf."))
@click.option('--interval',
              default=100,
              type=float,
              help="Delay between frames in milliseconds")
# Plotting options
@click.option('--x-label',
              help="The label on the x-axis.",
              show_default="name of the x-axis column")
@click.option('--y-label',
              required=False,
              help="The label on the y-axis.",
              show_default="no label")
@click.option('--x-logscale',
              is_flag=True,
              help="Set the x-axis to log scale.")
@click.option('--y-logscale',
              is_flag=True,
              help="Set the y-axis to log scale.")
@click.option('--x-bounds',
              type=float,
              nargs=2,
              help="The lower and upper bounds of the x-axis.")
@click.option('--y-bounds',
              type=float,
              nargs=2,
              help="The lower and upper bounds of the y-axis.")
@click.option('--title',
              '-t',
              help="Title of the graph.",
              show_default="subfile name")
@click.option(
    '--stylesheet',
    '-s',
    type=click.Path(exists=True, file_okay=True, dir_okay=False,
                    readable=True),
    envvar="SPECTRE_MPL_STYLESHEET",
    help=("Select a matplotlib stylesheet for customization of the plot, such "
          "as linestyle cycles, linewidth, fontsize, legend, etc. "
          "The stylesheet can also be set with the 'SPECTRE_MPL_STYLESHEET' "
          "environment variable."))
@click.option("--show-collocation-points", is_flag=True)
@click.option("--show-element-boundaries", is_flag=True)
@click.option("--show-basis-polynomials", is_flag=True)
def render_1d_command(h5_files, subfile_name, list_vars, vars, output, x_label,
                      y_label, x_logscale, y_logscale, x_bounds, y_bounds,
                      title, stylesheet, step, interval,
                      **plot_element_kwargs):
    """Render 1D data"""
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
        subfile_name = subfile_name[:-4]
    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name

    volfiles = [h5file.get_vol(subfile_name) for h5file in open_h5_files]
    dim = volfiles[0].get_dimension()
    assert dim == 1, (
        f"The selected subfile contains {dim}D volume data, not 1D.")
    obs_ids = volfiles[0].list_observation_ids()  # Already sorted by obs value
    all_vars = volfiles[0].list_tensor_components(obs_ids[0])
    if "InertialCoordinates_x" in all_vars:
        all_vars.remove("InertialCoordinates_x")

    # Print available variables and exit
    if list_vars:
        import rich.columns
        rich.print(rich.columns.Columns(all_vars))
        return
    for var in vars:
        if var not in all_vars:
            raise click.UsageError(f"Unknown variable '{var}'. "
                                   f"Available variables are: {all_vars}")
    if not vars:
        vars = {var: var for var in all_vars}
    plot_element_kwargs["vars"] = vars

    # Apply stylesheet
    if stylesheet is not None:
        plt.style.use(stylesheet)

    # Evaluate property cycles for each variable (by default this is just
    # 'color'). We do multiple plotting commands per variable (at least one per
    # element), so we don't want matplotlib to cycle through the properties at
    # every plotting command.
    prop_cycle = {
        key: cycle(values)
        for key, values in plt.rcParams['axes.prop_cycle'].by_key().items()
    }
    var_props = {
        var: {key: next(values)
              for key, values in prop_cycle.items()}
        for var in vars
    }
    plot_element_kwargs["var_props"] = var_props

    # Animate or single frame?
    if len(obs_ids) == 1:
        animate = False
        obs_id = obs_ids[0]
    elif step is None:
        animate = True
        obs_id = obs_ids[0]
    else:
        animate = False
        obs_id = obs_ids[step]
    obs_value = volfiles[0].get_observation_value(obs_id)

    # For Lagrange interpolation
    domain = deserialize_domain[1](volfiles[0].get_domain(obs_id))
    if domain.is_time_dependent():
        functions_of_time = deserialize_functions_of_time(
            volfiles[0].get_functions_of_time(obs_id))
        plot_element_kwargs["functions_of_time"] = functions_of_time
        plot_element_kwargs["time"] = obs_value

    # We store plots here so we can update them later
    plot_element_kwargs["handles"] = dict()

    # Plot first frame
    fig = plt.figure()
    for element, vars_data in iter_elements(volfiles, obs_id, vars):
        legend_items = plot_element(element, vars_data, **plot_element_kwargs)

    # Configure the axes
    if x_logscale:
        plt.xscale("log")
    if y_logscale:
        plt.yscale("log")
    plt.xlabel(x_label if x_label else "x")
    plt.ylabel(y_label)
    plt.legend(legend_items.values(), legend_items.keys())
    title_handle = plt.title(title if title else f"t = {obs_value:g}")
    if animate and not (x_bounds and y_bounds):
        data_bounds = get_bounds(volfiles, obs_ids, vars)
        if not x_bounds:
            x_bounds = data_bounds[0]
        if not y_bounds:
            y_bounds = data_bounds[1]
            if not y_logscale:
                margin = (y_bounds[1] - y_bounds[0]) * 0.05
                y_bounds[0] -= margin
                y_bounds[1] += margin
    if x_bounds:
        plt.xlim(*x_bounds)
    if y_bounds:
        plt.ylim(*y_bounds)

    if animate:
        import matplotlib.animation
        import rich.progress

        progress = rich.progress.Progress(
            rich.progress.TextColumn(
                "[progress.description]{task.description}"),
            rich.progress.BarColumn(), rich.progress.MofNCompleteColumn(),
            rich.progress.TimeRemainingColumn())
        task_id = progress.add_task("Rendering", total=len(obs_ids))

        def update(frame):
            obs_id = obs_ids[frame]
            obs_value = volfiles[0].get_observation_value(obs_id)
            if domain.is_time_dependent():
                functions_of_time = deserialize_functions_of_time(
                    volfiles[0].get_functions_of_time(obs_id))
                plot_element_kwargs["functions_of_time"] = functions_of_time
                plot_element_kwargs["time"] = obs_value
            title_handle.set_text(title if title else f"t = {obs_value:g}")
            for element, vars_data in iter_elements(volfiles, obs_id, vars):
                plot_element(element, vars_data, **plot_element_kwargs)
            progress.update(task_id, completed=frame + 1)

        anim = matplotlib.animation.FuncAnimation(fig=fig,
                                                  func=update,
                                                  frames=range(len(obs_ids)),
                                                  interval=interval,
                                                  blit=False)

    if output:
        if animate:
            if not output.endswith(".mp4"):
                output += ".mp4"
            with progress:
                anim.save(output, writer='ffmpeg')
        else:
            if not output.endswith(".pdf"):
                output += ".pdf"
            plt.savefig(output, bbox_inches="tight")
    else:
        if not os.environ.get("DISPLAY"):
            logger.warning(
                "No 'DISPLAY' environment variable is configured so plotting "
                "interactively is unlikely to work. Write the plot to a file "
                "with the --output/-o option.")
        plt.show()

    for h5file in open_h5_files:
        h5file.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    render_1d_command(help_option_name=["-h", "--help"])
