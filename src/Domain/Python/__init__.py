# Distributed under the MIT License.
# See LICENSE.txt for details.

from ._Pybindings import *

Block = {1: Block1D, 2: Block2D, 3: Block3D}
Domain = {1: Domain1D, 2: Domain2D, 3: Domain3D}
ElementId = {1: ElementId1D, 2: ElementId2D, 3: ElementId3D}

deserialize_domain = {
    1: deserialize_domain_1d,
    2: deserialize_domain_2d,
    3: deserialize_domain_3d,
}
