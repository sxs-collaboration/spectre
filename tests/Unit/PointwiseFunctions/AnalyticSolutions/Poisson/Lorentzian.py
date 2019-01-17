# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x):
    return 1. / np.sqrt(1. + np.dot(x, x))


def source(x):
    return 3. / np.sqrt(1. + np.dot(x, x))**5


def auxiliary_source(x):
    return np.zeros(len(x))
