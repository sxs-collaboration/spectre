# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import multiprocessing
from typing import Iterable, Sequence, Union

import click
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr
from spectre.Domain import (
    ElementId,
    block_logical_coordinates,
    deserialize_domain,
    deserialize_functions_of_time,
    element_logical_coordinates,
)
from spectre.Interpolation import Irregular
from spectre.Spectral import Mesh

logger = logging.getLogger(__name__)


def interpolate_to_coords(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_id: int,
    tensor_components: Sequence[str],
    target_coords: Union[np.ndarray, tnsr.I[DataVector, 3]],
) -> np.ndarray:
    """Interpolate volume data to target coordinates.

    Arguments:
      volfiles: Open volume data files.
      obs_id: Observation ID present in all volume data files.
      tensor_components: List of tensor components to interpolate. Each string
        specified here must be a dataset name in the volume files.
      target_coords: Coordinates to interpolate to. Must be an array of shape
        (num_target_points, dim) or a 'tnsr.I[DataVector, 3]'.

    Returns:
      Array of shape (len(tensor_components), num_target_points) with the
      interpolated data at all 'target_coords'.
    """
    if isinstance(target_coords, np.ndarray):
        num_target_points, dim = target_coords.shape
        target_coords = tnsr.I[DataVector, 3](target_coords.T)
    else:
        dim = target_coords.size
        num_target_points = len(target_coords[0])
    # We'll fill this array and return it
    interpolated_data = np.empty((len(tensor_components), num_target_points))
    interpolated_data.fill(np.nan)
    # This is to keep track of completed target points to terminate early
    filled_data = np.zeros(num_target_points, dtype=bool)
    # We'll read these from the first volfile
    domain = None
    time = None
    functions_of_time = None
    block_logical_coords = None
    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    for volfile in volfiles:
        assert dim == volfile.get_dimension(), (
            f"'target_coords' has dimension {dim} but volume data has "
            f"dimension {volfile.get_dimension()}."
        )
        if domain is None:
            domain = deserialize_domain[dim](volfile.get_domain(obs_id))
            if domain.is_time_dependent():
                time = volfile.get_observation_value(obs_id)
                functions_of_time = deserialize_functions_of_time(
                    volfile.get_functions_of_time(obs_id)
                )
            block_logical_coords = block_logical_coordinates(
                domain=domain,
                inertial_coords=target_coords,
                time=time,
                functions_of_time=functions_of_time,
            )
        all_grid_names = volfile.get_grid_names(obs_id)
        all_element_ids = list(map(ElementId[dim], all_grid_names))
        all_extents = volfile.get_extents(obs_id)
        all_bases = volfile.get_bases(obs_id)
        all_quadratures = volfile.get_quadratures(obs_id)
        meshes = {
            mesh_args[0]: Mesh[dim](*mesh_args[1:])
            for mesh_args in zip(
                all_element_ids, all_extents, all_bases, all_quadratures
            )
        }
        # Pre-load the tensor data because it's stored contiguously for all
        # grids in the file
        tensor_data = np.asarray(
            [
                volfile.get_tensor_component(obs_id, component).data
                for component in tensor_components
            ]
        )
        # Map the target points to element-logical coordinates
        element_logical_coords = element_logical_coordinates(
            all_element_ids, block_logical_coords
        )
        for element_id, point in element_logical_coords.items():
            offset, length = spectre_h5.offset_and_length_for_grid(
                str(element_id), all_grid_names, all_extents
            )
            element_data = tensor_data[:, offset : offset + length]
            interpolant = Irregular[dim](
                source_mesh=meshes[element_id],
                target_logical_coords=point.element_logical_coords,
            )
            interpolated_data[:, point.offsets] = np.asarray(
                [
                    interpolant.interpolate(DataVector(component))
                    for component in element_data
                ]
            )
            filled_data[point.offsets] = True
        # Terminate early if all data has been filled
        if filled_data.all():
            break
    return interpolated_data


def parse_points(ctx, param, values):
    if not values:
        return None
    points = [list(map(float, value.split(","))) for value in values]
    dim = len(points[0])
    if any([len(point) != dim for point in points]):
        raise click.BadParameter(
            "All specified points must have the same dimension"
        )
    return np.array(points)


@click.command()
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help=(
        "Name of subfile within h5 file containing volume data to interpolate."
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
    "vars",
    multiple=True,
    help=(
        "Variables to interpolate. List any tensor components "
        "in the volume data file, such as 'Shift_x'."
    ),
)
@click.option(
    "--target-coords-file",
    "-t",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help=(
        "Text file with target coordinates to interpolate to. "
        "Must have 'dim' columns with Cartesian coordinates. "
        "Rows enumerate points. "
        "Can be the output of 'numpy.savetxt'."
    ),
)
@click.option(
    "--target-coords",
    "-p",
    multiple=True,
    callback=parse_points,
    help=(
        "List target coordinates explicitly, e.g. '0,0,0'. "
        "Can be specified multiple times to quickly interpolate "
        "to a couple of target points."
    ),
)
@click.option(
    "--step",
    type=int,
    help=(
        "Observation step number. "
        "Specify '-1' for the last step in the file. "
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
    "--output", "-o", type=click.Path(writable=True), help="Output text file"
)
@click.option(
    "--delimiter",
    default=None,
    show_default="whitespace",
    help=(
        "Delimiter separating columns for both the "
        "'--target-coords-file' and the '--output' file."
    ),
)
def interpolate_to_coords_command(
    h5_files,
    subfile_name,
    list_vars,
    vars,
    target_coords,
    target_coords_file,
    output,
    step,
    time,
    delimiter,
):
    """Interpolate volume data to target coordinates."""
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
    obs_ids = volfiles[0].list_observation_ids()
    obs_values = list(map(volfiles[0].get_observation_value, obs_ids))
    dim = volfiles[0].get_dimension()

    # Select observation
    if (step is None) == (time is None):
        raise click.UsageError(
            f"Specify either '--step' (in [0, {len(obs_ids) - 1}], or -1) or "
            f"'--time' (in [{obs_values[0]:g}, {obs_values[-1]:g}])."
        )
    if step is None:
        # Find closest observation to specified time
        step = np.argmin(np.abs(time - np.array(obs_values)))
        obs_value = obs_values[step]
        if obs_value != time:
            logger.info(
                f"Selected closest observation to t = {time}: "
                f"step {step} at t = {obs_value:g}"
            )
    obs_id = obs_ids[step]

    # Print available variables and exit
    all_vars = volfiles[0].list_tensor_components(obs_id)
    if list_vars or not vars:
        import rich.columns

        rich.print(rich.columns.Columns(all_vars))
        return
    for var in vars:
        if var not in all_vars:
            raise click.UsageError(
                f"Unknown variable '{var}'. Available variables are: {all_vars}"
            )

    # Load target coords from file
    if (target_coords is None) == (target_coords_file is None):
        raise click.UsageError(
            "Specify either '--target-coords' / '-p' or "
            "'--target-coords-file' / '-t'."
        )
    if target_coords_file:
        target_coords = np.loadtxt(
            target_coords_file, ndmin=2, delimiter=delimiter
        )

    # Interpolate!
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        disable=(len(volfiles) == 1),
    )
    task_id = progress.add_task("Interpolating files")
    volfiles_progress = progress.track(volfiles, task_id=task_id)
    with progress:
        interpolated_data = interpolate_to_coords(
            volfiles_progress,
            target_coords=target_coords,
            obs_id=obs_id,
            tensor_components=vars,
        )
        progress.update(task_id, completed=len(volfiles))

    # Output result
    column_names = ["X", "Y", "Z"][:dim] + list(vars)
    column_data = np.hstack([target_coords, interpolated_data.T])
    if output:
        np.savetxt(
            output,
            column_data,
            delimiter=delimiter or " ",
            header=(
                f"t = {obs_values[step]:g}\n"
                + (delimiter or " ").join(column_names)
            ),
        )
    else:
        import rich.table

        table = rich.table.Table(*column_names, box=None)
        for row in column_data:
            table.add_row(*list(map(str, row)))
        rich.print(table)
        rich.print(f"(t = {obs_values[step]:g})")


if __name__ == "__main__":
    interpolate_to_coords_command(help_option_names=["-h", "--help"])
