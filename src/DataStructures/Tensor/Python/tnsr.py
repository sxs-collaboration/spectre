# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Defines the 'Tensor.tnsr' for shortcuts to Tensor classes

Use this module like this:

    from spectre.DataStructures.Tensor import tnsr, Frame, Scalar
    from spectre.DataStructures import DataVector

    scalar = Scalar[DataVector]()
    vector = tnsr.I[DataVector, 3, Frame.Inertial]()
"""

from spectre.DataStructures.Tensor import TensorMeta

i = TensorMeta("i")
I = TensorMeta("I")
ij = TensorMeta("ij")
ii = TensorMeta("ii")
II = TensorMeta("II")
ijj = TensorMeta("ijj")
