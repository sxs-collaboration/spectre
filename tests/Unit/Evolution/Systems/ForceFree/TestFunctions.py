# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Fluxes import levi_civita_symbol


# Functions for testing SpatialCurrentDensity
def spatial_current_density(tilde_q, tilde_e, tilde_b, eta,
                            sqrt_det_spatial_metric, spatial_metric):
    charge_density = tilde_q / sqrt_det_spatial_metric
    electric_field = tilde_e / sqrt_det_spatial_metric
    magnetic_field = tilde_b / sqrt_det_spatial_metric

    electric_field_one_form = np.einsum("a, ia", electric_field,
                                        spatial_metric)
    magnetic_field_one_form = np.einsum("a, ia", magnetic_field,
                                        spatial_metric)

    electric_field_squared = np.einsum("a, a", electric_field_one_form,
                                       electric_field)
    magnetic_field_squared = np.einsum("a, a", magnetic_field_one_form,
                                       magnetic_field)
    electric_field_dot_magnetic_field = np.einsum("a, a",
                                                  electric_field_one_form,
                                                  magnetic_field)

    result = tilde_e * 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                e_ijk = levi_civita_symbol(i, j, k) / sqrt_det_spatial_metric
                # drift current term
                result[i] += charge_density * e_ijk * electric_field_one_form[
                    j] * magnetic_field_one_form[k] / magnetic_field_squared
        # parallel current terms
        result[i] += eta * electric_field_dot_magnetic_field * magnetic_field[
            i] / magnetic_field_squared
        result[i] += eta * max(
            electric_field_squared - magnetic_field_squared,
            0.0) * electric_field[i] / magnetic_field_squared

    return result


# end functions for testing SpatialCurrentDensity
