# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x):
    return 1. / np.sqrt(1. + np.dot(x, x))


def field_gradient(x):
    return -np.asarray(x) / np.sqrt(1. + np.dot(x, x))**3


def field_flux(x):
    return field_gradient(x)


def source(x):
    return 3. / np.sqrt(1. + np.dot(x, x))**5
