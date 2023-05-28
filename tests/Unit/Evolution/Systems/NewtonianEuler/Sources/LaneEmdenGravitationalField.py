# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from PointwiseFunctions.AnalyticSolutions.NewtonianEuler.LaneEmdenStar import (
    gravitational_field,
    mass_density,
)


def source_momentum_density(
    mass_density, momentum_density, x, central_mass_density, polytropic_constant
):
    g = gravitational_field(x, central_mass_density, polytropic_constant)
    return mass_density * g


def source_energy_density(
    mass_density, momentum_density, x, central_mass_density, polytropic_constant
):
    g = gravitational_field(x, central_mass_density, polytropic_constant)
    return np.dot(momentum_density, g)
