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

a = TensorMeta("a")
A = TensorMeta("A")
ab = TensorMeta("ab")
aB = TensorMeta("aB")
Ab = TensorMeta("Ab")
aa = TensorMeta("aa")
AA = TensorMeta("AA")
abb = TensorMeta("abb")
Abb = TensorMeta("Abb")
aBcc = TensorMeta("aBcc")

i = TensorMeta("i")
I = TensorMeta("I")
ij = TensorMeta("ij")
iJ = TensorMeta("iJ")
Ij = TensorMeta("Ij")
ii = TensorMeta("ii")
II = TensorMeta("II")
ijj = TensorMeta("ijj")
iJJ = TensorMeta("iJJ")
Ijj = TensorMeta("Ijj")
iJkk = TensorMeta("iJkk")

iaa = TensorMeta("iaa")
