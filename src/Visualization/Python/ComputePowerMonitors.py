# Distributed under the MIT License.
# See LICENSE.txt for details.

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5 import TensorComponent, ElementVolumeData
from spectre.DataStructures import DataVector
from spectre import Spectral
import numpy as np


def compute_modal_coefs(source_file_name, subfile_name, tensor_name):
    """for each observation, for each element, calculates the modal coefficients
    u for an observed quantity and calculates the "marginalization" in each
    dimension x_i = \sqrt{ sum_{jk} u_{ijk})^2  / N_y N_z }. This is useful to
    estimate the accuracy of the simulation in each element.
    Parameters
    ----------
    source_file_name : str
        name of the .h5 file to open
    subfile_name : str
        name of the subfile name  e.g. /Volume
    tensor_name : str
        name of the tensor that is analyzed
    Returns
    -------
    A nested dictionary with keys observation id -> element name -> dimension
    which holds the modal coefficients of each observation, element
    and direction.
    """
    source_file = spectre_h5.H5File(source_file_name, "r")
    source_vol = source_file.get_vol(subfile_name)
    observations = sorted([obs for obs in source_vol.list_observation_ids()],
                          key=source_vol.get_observation_value)
    dim = source_vol.get_dimension()
    volume_data = {}
    for obs in observations:
        # the vols memory address may shift as we write to file,
        # so we need to get them every iteration
        source_file.close()
        source_vol = source_file.get_vol(subfile_name)
        extents = source_vol.get_extents(obs)
        bases = source_vol.get_bases(obs)
        quadratures = source_vol.get_quadratures(obs)
        grid_names = source_vol.get_grid_names(obs)

        # the raw tensor data
        res_tensor = np.array(source_vol.get_tensor_component(
            obs, tensor_name).data,
                              copy=False)
        volume_data[obs] = {}
        # iterate over elements
        for grid_name, extent, basis, quadrature in zip(
                grid_names, extents, bases, quadratures):
            mesh = Spectral.Mesh[dim](extent, basis, quadrature)
            mesh_slices = mesh.slices()
            #generate nodal to modal matrices in each direction
            nodal_to_modal_matrices = [
                np.asarray(Spectral.nodal_to_modal_matrix(slice))
                for slice in mesh_slices
            ]
            offset, length = spectre_h5.offset_and_length_for_grid(
                grid_name, grid_names, extents)
            # the tensor data of the current element reshaped into the mesh dims
            component_data = res_tensor[offset:offset + length].reshape(
                extent, order="F")
            # nodal to modal transformation of tensor data
            for l, matrix in enumerate(nodal_to_modal_matrices):
                component_data = np.apply_along_axis(
                    lambda x: np.dot(matrix, x), l, component_data)

            # create a dict for each element
            tensor_comps = {}

            dims = ["x", "y", "z"] if dim == 3 else ["x", "y"]
            # here we calculate x_i = \sqrt{ sum_{jk} u_{ijk})^2  / N_y N_z }
            # where u are the modal coefficients
            component_data *= component_data
            for i, direction in enumerate(dims):
                num_points_other_dims = np.prod(np.delete(extent, i))
                res_error = np.sqrt(
                    np.sum(component_data,
                           axis=tuple(k for k in range(dim) if k != i)) /
                    num_points_other_dims)
                tensor_comps[direction] = res_error
            volume_data[obs][grid_name] = tensor_comps

    source_file.close()
    return volume_data


def write_resolution_data(source_file_name,
                          source_subfile_name,
                          target_file_name,
                          target_subfile_name,
                          tensor_name,
                          coef_func,
                          copy_data=True):
    """creates a new .h5 file which shows the evolution of the estimated
    resolution of the system. This is done by computing the modal
    coefficients using `compute_modal_coefs`. The user then provides a
    function `coef_func` which takes these modal coefficients as input and
    returns an estimate for the resolution error in that direction, e.g.
    abs(coefs[-1]), i.e. the size of last coefficient or
    np.log10(coefs[0]) - np.log10(coefs[-1]) which shows how many orders of
    magnitude are spanned by the coefficients.
    However, care must be taken if filtering suppresses the
    last coefficients or symmetries cause odd/even coefficients to be zero. It
    is advised to plot the coefficients first to construct a sensible
    `coef_func`.

    Example usage for data from 32 nodes, parallelized over the files:

    def write_data(i):
        write_resolution_data(
            f"Volume{i}.h5", "/Volume",
            f"resolution{i}.h5", "/resolution", "Psi",
            lambda coefs: np.log10(coefs[0]) - np.log10(coefs[-2]))
    from multiprocessing import Pool
    with Pool() as p:
        p.map(write_data, range(32))

    Parameters
    ----------
    source_file_name : str
        name of the .h5 file to open
    source_subfile_name : str
        name of the subfile name in the source file e.g. /Volume
    target_file_name : str
        name of the .h5 file to write to
    target_subfile_name : str
        name of the subfile name in the target file  e.g. /resolution
    tensor_name : str
        name of the tensor that is analyzed
    coef_func : func
        A function that takes a 1D array of modal coefficients and returns a
        scalar which estimates the error of inside this element
    copy_data : bool
        Whether to copy over the tensor data into the target file
    """

    source_file = spectre_h5.H5File(source_file_name, "r")
    target_file = spectre_h5.H5File(target_file_name, "a")

    source_vol = source_file.get_vol(source_subfile_name)
    observations = sorted([obs for obs in source_vol.list_observation_ids()],
                          key=source_vol.get_observation_value)
    dim = source_vol.get_dimension()
    dims = ["x", "y", "z"] if dim == 3 else ["x", "y"]
    version = source_vol.get_version()
    source_file.close()
    # create a new target file where we write the resolution data
    target_file.insert_vol(target_subfile_name, version)
    target_file.close()
    modal_data = compute_modal_coefs(source_file_name, source_subfile_name,
                                     tensor_name)
    for obs in observations:
        # the vols memory address may shift as we write to file,
        # so we need to get them every iteration
        source_file.close()
        source_vol = source_file.get_vol(source_subfile_name)
        extents = source_vol.get_extents(obs)
        bases = source_vol.get_bases(obs)
        quadratures = source_vol.get_quadratures(obs)
        grid_names = source_vol.get_grid_names(obs)
        obs_value = source_vol.get_observation_value(obs)

        vol_data = []
        #the raw tensor data
        res_tensor = np.asarray(
            source_vol.get_tensor_component(obs, tensor_name).data)

        # we need to get and copy over the coordinate data to generate the xdmf
        # file
        coord_tensors = [
            np.asarray(
                source_vol.get_tensor_component(obs,
                                                "InertialCoordinates_x").data),
            np.asarray(
                source_vol.get_tensor_component(obs,
                                                "InertialCoordinates_y").data)
        ]
        if dim == 3:
            coord_tensors.append(
                np.asarray(
                    source_vol.get_tensor_component(
                        obs, "InertialCoordinates_z").data))

        # iterate over elements
        for grid_name, extent, basis, quadrature in zip(
                grid_names, extents, bases, quadratures):

            offset, length = spectre_h5.offset_and_length_for_grid(
                grid_name, grid_names, extents)

            scalar_comps = []

            for i, direction in enumerate(dims):
                estimated_error = coef_func(
                    modal_data[obs][grid_name][direction])
                scalar_path = f"resolution_{direction}"
                # we cant write scalar data to h5 yet, so we copy it over.
                scalar_comps.append(
                    TensorComponent(
                        scalar_path,
                        DataVector(np.full(np.prod(extent), estimated_error))))
                coord_path = f"InertialCoordinates_{direction}"
                scalar_comps.append(
                    TensorComponent(
                        coord_path,
                        DataVector(coord_tensors[i][offset:offset + length],
                                   copy=False)))
            if copy_data:
                scalar_comps.append(
                    TensorComponent(
                        f"{tensor_name}",
                        DataVector(res_tensor[offset:offset + length],
                                   copy=False)))

            vol_data.append(
                ElementVolumeData(grid_name, scalar_comps, extent, basis,
                                  quadrature))

        target_file.close()
        target_vol = target_file.get_vol(target_subfile_name)
        target_vol.write_volume_data(obs, obs_value, vol_data)
    source_file.close()
    target_file.close()
