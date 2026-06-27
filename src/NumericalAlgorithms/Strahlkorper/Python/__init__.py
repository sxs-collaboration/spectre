# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.DataStructures.Tensor import Frame

from ._Pybindings import *

Strahlkorper = {
    Frame.Grid: StrahlkorperGrid,
    Frame.Inertial: StrahlkorperInertial,
}
