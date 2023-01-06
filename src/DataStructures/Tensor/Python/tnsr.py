# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Defines the 'Tensor.tnsr' for shortcuts to Tensor classes

Use this module like this:

    from spectre.DataStructures.Tensor import tnsr, Frame, Scalar
    from spectre.DataStructures import DataVector

    scalar = Scalar[DataVector]()
    vector = tnsr.I[DataVector, 3, Frame.Inertial]()
"""

from spectre.DataStructures import Tensor, DataVector
from spectre.DataStructures.Tensor import Frame
import itertools


def dtype_to_name(dtype: type):
    if dtype is DataVector:
        return "DV"
    elif dtype is float:
        return "D"
    else:
        raise NotImplementedError


class TensorMeta:
    def __init__(self, name):
        self.__members__ = {
            (dtype, dim, frame):
            getattr(Tensor,
                    f"Tensor{name}{dtype_to_name(dtype)}{dim}{frame.name}")
            for dtype, dim, frame in itertools.product(
                [DataVector, float], [1, 2, 3],
                [Frame.ElementLogical, Frame.Inertial])
        }

    def _getitem(self, dtype: type, dim: int, frame: Frame = Frame.Inertial):
        return self.__members__[(dtype, dim, frame)]

    def __getitem__(self, key):
        try:
            return self._getitem(*key)
        except TypeError:
            return self._getitem(key)


I = TensorMeta("I")
