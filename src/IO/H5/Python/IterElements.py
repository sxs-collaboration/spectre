# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import spectre.IO.H5 as spectre_h5
from dataclasses import dataclass
from spectre.DataStructures.Tensor.EagerMath import determinant
from spectre.Domain import (ElementId, ElementMap, deserialize_domain,
                            deserialize_functions_of_time, FunctionOfTime)
from spectre.Domain.CoordinateMaps import (
    CoordinateMapElementLogicalToInertial1D,
    CoordinateMapElementLogicalToInertial2D,
    CoordinateMapElementLogicalToInertial3D)
from spectre.Spectral import Mesh, logical_coordinates
from typing import Union, Iterable, Optional, Dict

# functools.cached_property was added in Py 3.8. Fall back to a plain
# `property` in Py 3.7.
try:
    from functools import cached_property
except ImportError:
    cached_property = property


@dataclass(frozen=True)
class Element:
    id: Union[ElementId[1], ElementId[2], ElementId[3]]
    mesh: Union[Mesh[1], Mesh[2], Mesh[3]]
    map: Union[CoordinateMapElementLogicalToInertial1D,
               CoordinateMapElementLogicalToInertial2D,
               CoordinateMapElementLogicalToInertial3D]
    time: Optional[float]
    functions_of_time: Optional[Dict[str, FunctionOfTime]]
    # Offset and length in contiguous tensor data corresponding to this element
    data_slice: slice

    @property
    def dim(self):
        return self.mesh.dim

    @cached_property
    def logical_coordinates(self):
        return logical_coordinates(self.mesh)

    @cached_property
    def inertial_coordinates(self):
        return self.map(self.logical_coordinates, self.time,
                        self.functions_of_time)

    @cached_property
    def inv_jacobian(self):
        return self.map.inv_jacobian(self.logical_coordinates, self.time,
                                     self.functions_of_time)

    @cached_property
    def jacobian(self):
        return self.map.jacobian(self.logical_coordinates, self.time,
                                 self.functions_of_time)

    @cached_property
    def det_jacobian(self):
        return determinant(self.jacobian)


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
        domain = deserialize_domain[dim](volfile.get_domain(obs_id))
        if domain.is_time_dependent():
            time = volfile.get_observation_value(obs_id)
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id))
        else:
            time = None
            functions_of_time = None
        all_grid_names = volfile.get_grid_names(obs_id)
        all_element_ids = [ElementId[dim](name) for name in all_grid_names]
        all_extents = volfile.get_extents(obs_id)
        all_bases = volfile.get_bases(obs_id)
        all_quadratures = volfile.get_quadratures(obs_id)
        all_meshes = [
            Mesh[dim](*mesh_args)
            for mesh_args in zip(all_extents, all_bases, all_quadratures)
        ]
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
            data_slice = slice(offset, offset + length)
            element_map = ElementMap(element_id, domain)
            element = Element(element_id,
                              mesh=mesh,
                              map=element_map,
                              time=time,
                              functions_of_time=functions_of_time,
                              data_slice=data_slice)
            if tensor_components:
                yield element, tensor_data[:, offset:offset + length]
            else:
                yield element
