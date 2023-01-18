# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import spectre.IO.H5 as spectre_h5
from dataclasses import dataclass
from spectre.Domain import ElementId
from spectre.Spectral import Mesh
from typing import Union, Iterable, Optional


@dataclass
class Element:
    id: Union[ElementId[1], ElementId[2], ElementId[3]]
    mesh: Union[Mesh[1], Mesh[2], Mesh[3]]
    inertial_coords: np.ndarray


def iter_elements(volfiles: Union[spectre_h5.H5Vol,
                                  Iterable[spectre_h5.H5Vol]],
                  obs_id: int,
                  tensor_components: Optional[Iterable[str]] = None):
    """Return volume data by element

    Arguments:
      volfiles: Open spectre H5 volume files. Can be a single volfile or a list,
        but can also be an iterator that opens and closes the files on demand.
      obs_id: Observation ID
      tensor_components: Tensor components to retrieve. Can be empty.

    Returns: Iterator over all elements in all 'volfiles'. Yields either just
      the 'Element' with structural information if 'tensor_components' is
      empty, or both the 'Element' and an 'np.ndarray' with the tensor data
      listed in 'tensor_components'. The tensor data has shape
      `(len(tensor_components), num_points)`.
    """
    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    for volfile in volfiles:
        dim = volfile.get_dimension()
        all_grid_names = volfile.get_grid_names(obs_id)
        all_element_ids = list(map(ElementId[dim], all_grid_names))
        all_extents = volfile.get_extents(obs_id)
        all_bases = volfile.get_bases(obs_id)
        all_quadratures = volfile.get_quadratures(obs_id)
        all_meshes = [
            Mesh[dim](*mesh_args)
            for mesh_args in zip(all_extents, all_bases, all_quadratures)
        ]
        # Get coordinates
        inertial_coords = np.array([
            volfile.get_tensor_component(obs_id,
                                         "InertialCoordinates" + xyz).data
            for xyz in ["_x", "_y", "_z"][:dim]
        ])
        # Pre-load the tensor data because it's stored contiguously for all
        # grids in the file
        if tensor_components:
            tensor_data = np.asarray([
                volfile.get_tensor_component(obs_id, component).data
                for component in tensor_components
            ])
        # Iterate elements in this file
        for grid_name, element_id, mesh in zip(all_grid_names, all_element_ids,
                                               all_meshes):
            offset, length = spectre_h5.offset_and_length_for_grid(
                grid_name, all_grid_names, all_extents)
            element = Element(
                element_id,
                mesh,
                inertial_coords=inertial_coords[:, offset:offset + length])
            if tensor_components:
                yield element, tensor_data[:, offset:offset + length]
            else:
                yield element
