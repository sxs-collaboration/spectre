# Distributed under the MIT License.
# See LICENSE.txt for details.

from typing import Sequence, Type

from ._Pybindings import *


def interpolate_tensors_to_points(
    *args,
    tensor_names: Sequence[str],
    tensor_types: Sequence[Type],
    **kwargs,
):
    """Wrapper around 'interpolate_to_points' for tensors.

    Interpolates volume data to points by calling 'interpolate_to_points'.
    However, instead of passing a list of tensor components
    (like 'SpatialMetric_xx') you can pass a list of tensor names (like
    'SpatialMetric') and their corresponding types (like
    'tnsr.ii[DataVector, 3]') and this function will assemble the tensors
    and return them as a list.

    Note: It would be nice to move this function into C++, but that's not
    trivial because the tensor types are not known at compile time. This can be
    revisited when we have Python support for Variables.
    """
    from spectre.DataStructures import DataVector

    tensors = {
        tensor_name: tensor_type()
        for tensor_name, tensor_type in zip(tensor_names, tensor_types)
    }
    tensor_components = []
    for name, tensor in tensors.items():
        tensor_components += [
            name + tensor.component_suffix(i) for i in range(tensor.size)
        ]
    data = interpolate_to_points(
        *args,
        **kwargs,
        tensor_components=tensor_components,
    )
    j = 0
    for tensor in tensors.values():
        for i in range(tensor.size):
            tensor[i] = DataVector(data[j])
            j += 1
    return tensors.values()
