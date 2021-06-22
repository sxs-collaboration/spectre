# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import (DataVector, TensorComponent,
                                    ElementVolumeData)
from spectre import Spectral, Interpolation
import numpy as np
import sys
from multiprocessing import Pool
import argparse
import re
import glob
import logging

basis_dict = {
    "Legendre": Spectral.Basis.Legendre,
    "Chebyshev": Spectral.Basis.Chebyshev,
    "FiniteDifference": Spectral.Basis.FiniteDifference
}
quadrature_dict = {
    "Gauss": Spectral.Quadrature.Gauss,
    "GaussLobatto": Spectral.Quadrature.GaussLobatto,
    "CellCentered": Spectral.Quadrature.CellCentered,
    "FaceCentered": Spectral.Quadrature.FaceCentered
}


def basis_from_string(names):
    return (basis_dict[names] if isinstance(names, str) else
            [basis_dict[name] for name in names])


def quadrature_from_string(names):
    return (quadrature_dict[names] if isinstance(names, str) else
            [quadrature_dict[name] for name in names])


# Function to forward arguments to the multiprocessing Pool. We can not use a
# lambda because the function needs to be pickled.
# In python 3 we can switch to `starmap`.
def forward_args(args):
    interpolate_h5_file(**args)


Mesh = {1: Spectral.Mesh1D, 2: Spectral.Mesh2D, 3: Spectral.Mesh3D}

RegularGrid = {
    1: Interpolation.RegularGrid1D,
    2: Interpolation.RegularGrid2D,
    3: Interpolation.RegularGrid3D
}


def interpolate_h5_file(source_file_path,
                        target_mesh,
                        target_file_path,
                        source_volume_data,
                        target_volume_data,
                        components_to_interpolate=None,
                        obs_start=-np.inf,
                        obs_end=np.inf,
                        obs_stride=1):
    """Interpolates an h5 file to a desired grid

    The function reads data from `source_volume_data` inside `source_file_path`,
    interpolates all components specified by `components_to_interpolate` to the
    grid specified by `target_mesh` and writes the results into
    `target_volume_data` inside `target_file_path`. The `target_file_path` can
    be the same as the `source_file_path` if the volume subfile paths are
    different.

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
        can be found. Requires leading `/` but no `.vol` file extension.
    target_volume_data: str, optional
        the name of the .vol file inside the target file where the target data
        is written. Requires leading `/` but no `.vol` file extension.
    obs_start: float, optional
        disregards all observations with observation value strictly before
        `obs_start`
    obs_end: float, optional
        disregards all observations with observation value strictly after
        `obs_end`
    obs_stride: float, optional
        will only take every `obs_stride` observation
    """

    if target_file_path == source_file_path:
        if source_volume_data == target_volume_data:
            raise NameError(
                "If the source and target files are the same, "
                "the source and target volume_data need to be different.")
        source_file = spectre_h5.H5File(source_file_path, "r+")
        target_file = source_file
    else:
        source_file = spectre_h5.H5File(source_file_path, "r")
        target_file = spectre_h5.H5File(target_file_path, "a")

    source_vol = source_file.get_vol(source_volume_data)
    dim = source_vol.get_dimension()
    # apply observation filter
    observations = [
        obs for obs in source_vol.list_observation_ids()
        if obs_start <= source_vol.get_observation_value(obs) <= obs_end
    ][::obs_stride]

    target_file.insert_vol(target_volume_data, source_vol.get_version())

    for obs in observations:
        # the vols memory address may shift as we write to file,
        # so we need to get them every iteration
        source_vol = source_file.get_vol(source_volume_data)
        extents = source_vol.get_extents(obs)
        bases = source_vol.get_bases(obs)
        quadratures = source_vol.get_quadratures(obs)
        tensor_names = source_vol.list_tensor_components(obs)
        grid_names = source_vol.get_grid_names(obs)
        obs_value = source_vol.get_observation_value(obs)

        if components_to_interpolate is not None:
            tensor_names = list(
                set(tensor_name for pattern in components_to_interpolate
                    for tensor_name in tensor_names
                    if re.match(pattern, tensor_name)))

        # pre-load all tensors to avoid loading the full tensor for each element
        tensors = [
            np.array(source_vol.get_tensor_component(obs, name), copy=False)
            for name in tensor_names
        ]

        volume_data = []
        # iterate over elements
        for grid_name, extent, basis, quadrature in zip(
                grid_names, extents, bases, quadratures):
            source_mesh = Mesh[dim](extent, basis_from_string(basis),
                                    quadrature_from_string(quadrature))

            interpolant = RegularGrid[dim](source_mesh, target_mesh)

            tensor_comps = []
            offset, length = spectre_h5.offset_and_length_for_grid(
                grid_name, grid_names, extents)
            # iterate over tensors
            for j, tensor in enumerate(tensors):
                component_data = DataVector(tensor[offset:offset + length],
                                            copy=False)
                interpolated_tensor = interpolant.interpolate(component_data)
                tensor_path = "{}/{}".format(grid_name, tensor_names[j])
                tensor_comps.append(
                    TensorComponent(
                        tensor_path, DataVector(interpolated_tensor,
                                                copy=False)))

            volume_data.append(
                ElementVolumeData(target_mesh.extents(), tensor_comps,
                                  target_mesh.basis(),
                                  target_mesh.quadrature()))
        target_vol = target_file.get_vol(target_volume_data)
        target_vol.write_volume_data(obs, obs_value, volume_data)

    source_file.close()
    if not target_file is source_file:
        target_file.close()


def parse_args(sys_args):
    """
    defines and parses the command line arguments

    Parameters
    ----------
    sys_args arguments passed via command line

    Returns
    -------
    A dictionary of the parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=
        "Interpolate the tensor components of h5 files to a desired grid")

    parser.add_argument(
        "--source-file-prefix",
        required=True,
        help=(
            "The prefix for the .h5 source files. All files starting with the "
            "prefix followed by a number will be interpolated."))

    parser.add_argument("--source-subfile-name",
                        required=True,
                        help=("The name of the volume data subfile within the "
                              "source files in which the data is contained"))

    parser.add_argument(
        "--target-file-prefix",
        default=None,
        help=
        ("The prefix for the target files where the interpolated data is "
         "written. When no target file is specified, the interpolated data is "
         "written to the corresponding source file in a new volume data "
         "subfile."))

    parser.add_argument(
        "--target-subfile-name",
        required=True,
        help=("The name of the volume data subfile within the target "
              "files where the data will be written."))

    parser.add_argument(
        "--tensor-components",
        nargs='*',
        help=("The names of the tensors that are to be interpolated. "
              "Accepts regular expression. "
              "If none are specified, all tensors are interpolated."))

    parser.add_argument(
        "--target-extents",
        type=int,
        nargs='+',
        required=True,
        help=("The extents of the target grid. "
              "Can be different for each dimension e.g. '3 5 4'"))

    parser.add_argument("--target-basis",
                        nargs='+',
                        choices=basis_dict.keys(),
                        required=True,
                        help=("The basis of the target grid. "
                              "Can be different for each dimension"))

    parser.add_argument("--target-quadrature",
                        nargs='+',
                        choices=quadrature_dict.keys(),
                        required=True,
                        help=("The quadrature of the target grid. "
                              "Can be different for each dimension"))

    parser.add_argument(
        "--start-time",
        type=float,
        default=-np.inf,
        help=("Disregard all observations with value before this point"))

    parser.add_argument(
        "--stop-time",
        type=float,
        default=np.inf,
        help=("Disregard all observations with value after this point"))

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=("Stride through observations with this step size."))

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help=("The maximum number of processes to be started. "
              "A process is spawned for each source file up to this number."))

    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Verbosity (-v, -vv, ...)")

    return parser.parse_args(sys_args)


if __name__ == "__main__":

    parsed_args = parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.WARNING - parsed_args.verbose * 10)
    source_files = glob.glob(parsed_args.source_file_prefix + "[0-9]*.h5")
    file_numbers = [
        re.match(parsed_args.source_file_prefix + "([0-9]*).h5",
                 source_file).group(1) for source_file in source_files
    ]
    target_files = [
        "{}{}.h5".format(parsed_args.target_file_prefix, number)
        for number in file_numbers
    ]

    if len(source_files) == 0:
        raise NameError("No files found matching the input pattern.")

    # construct target mesh
    target_basis = basis_from_string(parsed_args.target_basis) if len(
        parsed_args.target_basis) > 1 else basis_from_string(
            parsed_args.target_basis[0])
    target_quadrature = quadrature_from_string(
        parsed_args.target_quadrature) if len(
            parsed_args.target_quadrature) > 1 else quadrature_from_string(
                parsed_args.target_quadrature[0])
    target_extents = parsed_args.target_extents[0] if len(
        parsed_args.target_extents) == 1 else parsed_args.target_extents

    # get dimension from first file
    source_file = spectre_h5.H5File(source_files[0], "a")
    source_vol = source_file.get_vol(parsed_args.source_subfile_name)
    dim = source_vol.get_dimension()
    target_mesh = Mesh[dim](target_extents, target_basis, target_quadrature)

    interpolate_args = []
    logging.info("Source and target files/volumes are as follows:")

    for source_file, target_file in zip(source_files, target_files):
        interpolate_args.append(
            dict(source_file_path=source_file,
                 target_mesh=target_mesh,
                 target_file_path=target_file,
                 source_volume_data=parsed_args.source_subfile_name,
                 target_volume_data=parsed_args.target_subfile_name,
                 components_to_interpolate=parsed_args.tensor_components,
                 obs_start=parsed_args.start_time,
                 obs_end=parsed_args.stop_time,
                 obs_stride=parsed_args.stride))

        logging.info("{}{} => {}{}".format(source_file,
                                           parsed_args.source_subfile_name,
                                           target_file,
                                           parsed_args.target_subfile_name))

    # change to p.starmap in python3
    with Pool(parsed_args.jobs) as p:
        p.map(forward_args, interpolate_args)
