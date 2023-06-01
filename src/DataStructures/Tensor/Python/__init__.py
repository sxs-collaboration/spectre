# Distributed under the MIT License.
# See LICENSE.txt for details.

# Define Tensor type lookup tables
import itertools

from spectre.DataStructures import DataVector

from ._Pybindings import *
from .Frame import Frame


def _dtype_to_name(dtype: type):
    if dtype is DataVector:
        return "DV"
    elif dtype is float:
        return "D"
    else:
        raise NotImplementedError


class TensorMeta:
    def __init__(self, name: str):
        self.name = name

    def _getitem(self, dtype: type, dim: int, frame: Frame = Frame.Inertial):
        return globals()[
            f"Tensor{self.name}{_dtype_to_name(dtype)}{dim}{frame.name}"
        ]

    def __getitem__(self, key):
        try:
            return self._getitem(*key)
        except TypeError:
            return self._getitem(key)


class JacobianMeta(TensorMeta):
    def __init__(self, inverse: bool):
        self.inverse = inverse

    def _getitem(self, dtype: type, dim: int, frame: Frame = Frame.Inertial):
        return globals()[
            f"Jacobian{_dtype_to_name(dtype)}{dim}"
            + (
                f"{frame.name}ToElementLogical"
                if self.inverse
                else f"ElementLogicalTo{frame.name}"
            )
        ]


# Define Tensor types that aren't in 'tnsr.py'
Scalar = {DataVector: ScalarDV, float: ScalarD}
Jacobian = JacobianMeta(inverse=False)
InverseJacobian = JacobianMeta(inverse=True)

# Define a type annotation that means "any tensor". This should really be a
# common superclass of all Tensor types, but we currently don't have that in C++
# so we use a type alias to `typing.Any` as a workaround.
from typing import Any

Tensor = Any
