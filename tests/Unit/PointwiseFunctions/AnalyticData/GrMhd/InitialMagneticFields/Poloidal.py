# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def magnetic_field(
    x,
    pressure,
    sqrt_det_spatial_metric,
    dcoords_pressure,
    pressure_exponent,
    cutoff_pressure,
    vector_potential_amplitude,
):
    magnetic_field = np.zeros(x.shape)

    magnetic_field[0] = (
        -(
            vector_potential_amplitude
            * pressure_exponent
            / sqrt_det_spatial_metric
        )
        * (pressure - cutoff_pressure) ** (pressure_exponent - 1)
        * x[0]
        * dcoords_pressure[2]
    )

    magnetic_field[1] = (
        -(
            vector_potential_amplitude
            * pressure_exponent
            / sqrt_det_spatial_metric
        )
        * (pressure - cutoff_pressure) ** (pressure_exponent - 1)
        * x[1]
        * dcoords_pressure[2]
    )

    magnetic_field[2] = (
        vector_potential_amplitude / sqrt_det_spatial_metric
    ) * (
        2 * (pressure - cutoff_pressure) ** pressure_exponent
        + pressure_exponent
        * (pressure - cutoff_pressure) ** (pressure_exponent - 1)
        * (x[0] * dcoords_pressure[0] + x[1] * dcoords_pressure[1])
    )

    magnetic_field = np.where(pressure < cutoff_pressure, 0.0, magnetic_field)

    return magnetic_field
