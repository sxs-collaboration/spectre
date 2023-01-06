# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Defines the 'Frame' enum

'Frame' is a namespace with separate structs in C++. In Python, we represent it
as an enum instead. Since the frame is used as a template parameter in C++, it
is part of the name of bound classes in Python. The mapping for the 'Tensor'
class is done in `tnsr.py`.
"""

from enum import Enum, auto


class Frame(Enum):
    BlockLogical = auto()
    ElementLogical = auto()
    Grid = auto()
    Inertial = auto()
    Distorted = auto()
