# Distributed under the MIT License.
# See LICENSE.txt for details.

import fnmatch
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures.Tensor.EagerMath import determinant
from spectre.Domain import (
    ElementId,
    ElementMap,
    FunctionOfTime,
    deserialize_domain,
    deserialize_functions_of_time,
)
from spectre.Domain.CoordinateMaps import (
    CoordinateMapElementLogicalToInertial1D,
    CoordinateMapElementLogicalToInertial2D,
    CoordinateMapElementLogicalToInertial3D,
)
from spectre.Spectral import Mesh, logical_coordinates

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
    map: Union[
        CoordinateMapElementLogicalToInertial1D,
        CoordinateMapElementLogicalToInertial2D,
        CoordinateMapElementLogicalToInertial3D,
    ]
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
        return self.map(
            self.logical_coordinates, self.time, self.functions_of_time
        )

    @cached_property
    def inv_jacobian(self):
        return self.map.inv_jacobian(
            self.logical_coordinates, self.time, self.functions_of_time
        )

    @cached_property
    def jacobian(self):
        return self.map.jacobian(
            self.logical_coordinates, self.time, self.functions_of_time
        )

    @cached_property
    def det_jacobian(self):
        return determinant(self.jacobian)


def stripped_element_name(
    element_id: Union[str, ElementId[1], ElementId[2], ElementId[3]]
) -> bool:
    """The element name without the leading and trailing square bracket

    Square brackets have special meaning in glob patterns, so it's easier to
    strip them away when dealing with element names on the command line.
    """
    return str(element_id).strip("[]")


def include_element(
    element_id: Union[str, ElementId[1], ElementId[2], ElementId[3]],
    element_patterns: Optional[Sequence[str]],
) -> bool:
    """Whether or not the 'element_id' matches any of the 'element_patterns'

    The 'element_patterns' are interpreted as glob patterns.
    If 'element_patterns' is None, every element is included. If
    'element_patterns' is an empty list, no element is included.
    """
    if element_patterns is None:
        return True
    return any(
        fnmatch.fnmatch(stripped_element_name(element_id), element_pattern)
        for element_pattern in element_patterns
    )


def iter_elements(
    volfiles: Union[spectre_h5.H5Vol, Iterable[spectre_h5.H5Vol]],
    obs_ids: Optional[Union[int, Sequence[int]]],
    tensor_components: Optional[Iterable[str]] = None,
    element_patterns: Optional[Sequence[str]] = None,
):
    """Return volume data by element

    Arguments:
      volfiles: Open spectre H5 volume files. Can be a single volfile or a list,
        but can also be an iterator that opens and closes the files on demand.
      obs_id: An observation ID, a list of observation IDs, or None to iterate
        over all observation IDs.
      tensor_components: Tensor components to retrieve. Can be empty.
      element_patterns: If specified, include only elements that match any of
        these glob patterns. See 'IterElements.include_element' for details.

    Returns: Iterator over all elements in all 'volfiles'. Yields either just
      the 'Element' with structural information if 'tensor_components' is
      empty, or both the 'Element' and an 'np.ndarray' with the tensor data
      listed in 'tensor_components'. The tensor data has shape
      `(len(tensor_components), num_points)`.
    """
    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    if isinstance(obs_ids, int):
        obs_ids = [obs_ids]
    # Assuming the domain is the same in all volfiles at all observations to
    # speed up the script
    domain = None
    for volfile in volfiles:
        all_obs_ids = volfile.list_observation_ids()
        if obs_ids:
            selected_obs_ids = [
                obs_id for obs_id in obs_ids if obs_id in all_obs_ids
            ]
        else:
            selected_obs_ids = all_obs_ids
        dim = volfile.get_dimension()
        for obs_id in selected_obs_ids:
            # Filter by element patterns first to avoid doing unnecessary work
            # if the volfile doesn't contain any of the requested elements
            all_grid_names = volfile.get_grid_names(obs_id)
            if element_patterns is not None:
                grid_names = [
                    grid_name
                    for grid_name in all_grid_names
                    if include_element(grid_name, element_patterns)
                ]
            else:
                grid_names = all_grid_names
            if not grid_names:
                continue
            element_ids = [ElementId[dim](name) for name in grid_names]
            # Reconstruct meshes
            all_extents = volfile.get_extents(obs_id)
            all_bases = volfile.get_bases(obs_id)
            all_quadratures = volfile.get_quadratures(obs_id)
            meshes = [
                Mesh[dim](extents, bases, quadratures)
                for grid_name, extents, bases, quadratures in zip(
                    all_grid_names, all_extents, all_bases, all_quadratures
                )
                if grid_name in grid_names
            ]
            # Deserialize domain and functions of time
            if not domain:
                domain = deserialize_domain[dim](volfile.get_domain(obs_id))
            time = volfile.get_observation_value(obs_id)
            if domain.is_time_dependent():
                functions_of_time = deserialize_functions_of_time(
                    volfile.get_functions_of_time(obs_id)
                )
            else:
                functions_of_time = None
            # Pre-load the tensor data because it's stored contiguously for all
            # grids in the file
            if tensor_components:
                tensor_data = np.asarray(
                    [
                        volfile.get_tensor_component(obs_id, component).data
                        for component in tensor_components
                    ]
                )
            # Iterate elements in this file
            for grid_name, element_id, mesh in zip(
                grid_names, element_ids, meshes
            ):
                offset, length = spectre_h5.offset_and_length_for_grid(
                    grid_name, all_grid_names, all_extents
                )
                data_slice = slice(offset, offset + length)
                element_map = ElementMap(element_id, domain)
                element = Element(
                    element_id,
                    mesh=mesh,
                    map=element_map,
                    time=time,
                    functions_of_time=functions_of_time,
                    data_slice=data_slice,
                )
                if tensor_components:
                    yield element, tensor_data[:, offset : offset + length]
                else:
                    yield element
