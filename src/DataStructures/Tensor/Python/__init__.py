# Distributed under the MIT License.
# See LICENSE.txt for details.

from ._PyTensor import *

from .Frame import Frame

# Define 'Scalar' type
from spectre.DataStructures import DataVector

Scalar = {DataVector: ScalarDV, float: ScalarD}
