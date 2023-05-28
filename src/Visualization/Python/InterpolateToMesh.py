# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import re
import sys
from multiprocessing import Pool

import click
import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre import Interpolation, Spectral
from spectre.DataStructures import DataVector
from spectre.IO.H5 import ElementVolumeData, TensorComponent

logger = logging.getLogger(__name__)


# Function to forward arguments to the multiprocessing Pool. We can not use a
# lambda because the function needs to be pickled. Note that `starmap` only
# works with an iterable, not with a dict.
def forward_kwargs(kwargs):
    interpolate_to_mesh(**kwargs)


def interpolate_to_mesh(
    source_file_path,
    target_mesh,
    target_file_path,
    source_volume_data,
    target_volume_data,
    components_to_interpolate=None,
    obs_start=-np.inf,
    obs_end=np.inf,
    obs_stride=1,
):
    """Interpolates an h5 file to a desired grid

    The function reads data from `source_volume_data` inside `source_file_path`,
    interpolates all components specified by `components_to_interpolate` to the
    grid specified by `target_mesh` and writes the results into
    `target_volume_data` inside `target_file_path`. The `target_file_path` can
    be the same as the `source_file_path` if the volume subfile paths are
    different.

    \f
    Parameters
    ----------
    source_file_path: str
        the path to the source file where the `source_volume_data` is
    target_mesh: spectre.Spectral.Mesh
        the mesh to which the data is interpolated
    components_to_interpolate: list of str, optional
        a list of all components that are to be interpolated. accepts regular
        expressions. By default ALL tensor components are interpolated.
    target_file_path: str, optional
        the path to where the interpolated data is written. By default this is
        set to `source_file_path` so the interpolated data is written to the
        same file, but in a different subfile specified by `target_volume_data`.
    source_volume_data: str, optional
        the name of the .vol file inside the source file where the source data
        can be found.
    target_volume_data: str, optional
        the name of the .vol file inside the target file where the target data
        is written.
    obs_start: float, optional
        disregards all observations with observation value strictly before
        `obs_start`
    obs_end: float, optional
        disregards all observations with observation value strictly after
        `obs_end`
    obs_stride: float, optional
        will only take every `obs_stride` observation
    """

    if source_volume_data.endswith(".vol"):
        source_volume_data = source_volume_data[:-4]
    if target_volume_data.endswith(".vol"):
        target_volume_data = target_volume_data[:-4]
    if not source_volume_data.startswith("/"):
        source_volume_data = "/" + source_volume_data
    if not target_volume_data.startswith("/"):
        target_volume_data = "/" + target_volume_data

    if target_file_path == source_file_path:
        if source_volume_data == target_volume_data:
            raise NameError(
                "If the source and target files are the same, "
                "the source and target volume_data need to be different."
            )
        source_file = spectre_h5.H5File(source_file_path, "r+")
        target_file = source_file
    else:
        source_file = spectre_h5.H5File(source_file_path, "r")
        target_file = spectre_h5.H5File(target_file_path, "a")

    source_vol = source_file.get_vol(source_volume_data)
    dim = source_vol.get_dimension()
    version = source_vol.get_version()
    # apply observation filter
    observations = [
        obs
        for obs in source_vol.list_observation_ids()
        if obs_start <= source_vol.get_observation_value(obs) <= obs_end
    ][::obs_stride]

    source_file.close_current_object()
    target_file.insert_vol(target_volume_data, version)
    target_file.close_current_object()

    for obs in observations:
        # the vols memory address may shift as we write to file,
        # so we need to get them every iteration
        source_file.close_current_object()
        source_vol = source_file.get_vol(source_volume_data)
        extents = source_vol.get_extents(obs)
        bases = source_vol.get_bases(obs)
        quadratures = source_vol.get_quadratures(obs)
        tensor_names = source_vol.list_tensor_components(obs)
        grid_names = source_vol.get_grid_names(obs)
        obs_value = source_vol.get_observation_value(obs)

        if components_to_interpolate:
            tensor_names = list(
                set(
                    tensor_name
                    for pattern in components_to_interpolate
                    for tensor_name in tensor_names
                    if re.match(pattern, tensor_name)
                )
            )

        # pre-load all tensors to avoid loading the full tensor for each element
        tensors = [
            np.array(
                source_vol.get_tensor_component(obs, name).data, copy=False
            )
            for name in tensor_names
        ]

        source_file.close_current_object()

        volume_data = []
        # iterate over elements
        for grid_name, extent, basis, quadrature in zip(
            grid_names, extents, bases, quadratures
        ):
            source_mesh = Spectral.Mesh[dim](extent, basis, quadrature)

            interpolant = Interpolation.RegularGrid[dim](
                source_mesh, target_mesh
            )

            tensor_comps = []
            offset, length = spectre_h5.offset_and_length_for_grid(
                grid_name, grid_names, extents
            )
            # iterate over tensors
            for j, tensor in enumerate(tensors):
                component_data = DataVector(
                    tensor[offset : offset + length], copy=False
                )
                interpolated_tensor = interpolant.interpolate(component_data)
                tensor_path = tensor_names[j]
                tensor_comps.append(
                    TensorComponent(
                        tensor_path, DataVector(interpolated_tensor, copy=False)
                    )
                )

            volume_data.append(
                ElementVolumeData(
                    element_name=grid_name,
                    components=tensor_comps,
                    extents=target_mesh.extents(),
                    basis=target_mesh.basis(),
                    quadrature=target_mesh.quadrature(),
                )
            )
        target_file.close_current_object()
        target_vol = target_file.get_vol(target_volume_data)
        target_vol.write_volume_data(obs, obs_value, volume_data)

    source_file.close()
    if not target_file is source_file:
        target_file.close()


@click.command(help=interpolate_to_mesh.__doc__)
@click.option(
    "--source-file-prefix",
    required=True,
    help=(
        "The prefix for the .h5 source files. All files starting with the "
        "prefix followed by a number will be interpolated."
    ),
)
@click.option(
    "--source-subfile-name",
    required=True,
    help=(
        "The name of the volume data subfile within the "
        "source files in which the data is contained"
    ),
)
@click.option(
    "--target-file-prefix",
    default=None,
    help=(
        "The prefix for the target files where the interpolated data is "
        "written. When no target file is specified, the interpolated data is "
        "written to the corresponding source file in a new volume data "
        "subfile."
    ),
)
@click.option(
    "--target-subfile-name",
    required=True,
    help=(
        "The name of the volume data subfile within the target "
        "files where the data will be written."
    ),
)
@click.option(
    "--tensor-component",
    "-t",
    multiple=True,
    help=(
        "The names of the tensors that are to be interpolated. "
        "Accepts regular expression. "
        "If none are specified, all tensors are interpolated."
    ),
)
@click.option(
    "--target-extents",
    callback=(
        lambda ctx, param, value: (
            list(map(int, value.split(","))) if value else []
        )
    ),
    required=True,
    help=(
        "The extents of the target grid, as a comma-separated list without "
        "spaces. Can be different for each dimension e.g. '3,5,4'"
    ),
)
@click.option(
    "--target-basis",
    type=click.Choice(Spectral.Basis.__members__),
    callback=lambda ctx, param, value: Spectral.Basis.__members__[value],
    required=True,
    help="The basis of the target grid.",
)
@click.option(
    "--target-quadrature",
    type=click.Choice(Spectral.Quadrature.__members__),
    callback=lambda ctx, param, value: Spectral.Quadrature.__members__[value],
    required=True,
    help="The quadrature of the target grid.",
)
@click.option(
    "--start-time",
    type=float,
    default=-np.inf,
    help="Disregard all observations with value before this point",
)
@click.option(
    "--stop-time",
    type=float,
    default=np.inf,
    help="Disregard all observations with value after this point",
)
@click.option(
    "--stride",
    type=int,
    default=1,
    help="Stride through observations with this step size.",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=None,
    help=(
        "The maximum number of processes to be started. "
        "A process is spawned for each source file up to this number."
    ),
)
def interpolate_to_mesh_command(
    source_file_prefix,
    source_subfile_name,
    target_file_prefix,
    target_subfile_name,
    target_extents,
    target_basis,
    target_quadrature,
    tensor_component,
    start_time,
    stop_time,
    stride,
    num_jobs,
):
    _rich_traceback_guard = True  # Hide traceback until here

    source_files = glob.glob(source_file_prefix + "[0-9]*.h5")
    file_numbers = [
        re.match(source_file_prefix + "([0-9]*).h5", source_file).group(1)
        for source_file in source_files
    ]
    target_files = [
        f"{target_file_prefix}{number}.h5" for number in file_numbers
    ]

    if len(source_files) == 0:
        raise NameError("No files found matching the input pattern.")

    dim = len(target_extents)
    target_mesh = Spectral.Mesh[dim](
        target_extents, target_basis, target_quadrature
    )

    interpolate_kwargs = []
    logger.info("Source and target files/volumes are as follows:")

    for source_file_path, target_file_path in zip(source_files, target_files):
        interpolate_kwargs.append(
            dict(
                source_file_path=source_file_path,
                target_mesh=target_mesh,
                target_file_path=target_file_path,
                source_volume_data=source_subfile_name,
                target_volume_data=target_subfile_name,
                components_to_interpolate=tensor_component,
                obs_start=start_time,
                obs_end=stop_time,
                obs_stride=stride,
            )
        )

        logger.info(
            "{}{} => {}{}".format(
                source_file_path,
                source_subfile_name,
                target_file_path,
                target_subfile_name,
            )
        )

    with Pool(num_jobs) as p:
        p.map(forward_kwargs, interpolate_kwargs)


if __name__ == "__main__":
    interpolate_volume_data_command(help_option_names=["-h", "--help"])
