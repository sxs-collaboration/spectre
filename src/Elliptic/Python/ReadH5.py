# Distributed under the MIT License.
# See LICENSE.txt for details.

from typing import Sequence, Union

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.IterElements import iter_elements


def read_matrix(volfiles: Union[spectre_h5.H5Vol, Sequence[spectre_h5.H5Vol]]):
    """Read matrix representation written by elliptic solver.

    Arguments:
      volfiles: Open spectre H5 volume files that contain volume data written
        by the 'BuildMatrix' phase of an elliptic solve.

    Returns: The matrix as a dense numpy array. If memory becomes a concern, we
      can change this to a sparse matrix.
    """
    if isinstance(volfiles, spectre_h5.H5Vol):
        volfiles = [volfiles]
    # Each observation is a column of the matrix
    obs_ids = volfiles[0].list_observation_ids()
    size = len(obs_ids)
    matrix = np.zeros((size, size))
    # Collect all variables to be able to order them
    all_tensor_components = volfiles[0].list_tensor_components(obs_ids[0])
    variables = sorted(
        [
            component
            for component in all_tensor_components
            if component.startswith("Variable_")
        ],
        key=lambda component: int(component.split("_")[1]),
    )
    # Collect all element IDs to be able to order them
    num_points_per_element = {
        element.id: element.mesh.number_of_grid_points()
        for element in iter_elements(volfiles, obs_ids=obs_ids[0])
    }
    element_ids = sorted(num_points_per_element.keys())
    slice_start_per_element = {}
    slice_start = 0
    for element_id in element_ids:
        slice_start_per_element[element_id] = slice_start
        slice_start += num_points_per_element[element_id] * len(variables)
    del num_points_per_element
    # Fill matrix
    for element, tensor_data in iter_elements(
        volfiles, obs_ids=None, tensor_components=variables
    ):
        # Column of the matrix is the observation ID
        col = int(element.time)
        # Slice of the row is determined by ordering of elements
        slice_length = np.prod(tensor_data.shape)
        slice_start = slice_start_per_element[element.id]
        matrix[
            slice_start : slice_start + slice_length,
            col,
        ] = np.concatenate(tensor_data)
    return matrix
