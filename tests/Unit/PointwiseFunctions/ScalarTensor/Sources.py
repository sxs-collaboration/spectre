# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from RampUpFunction import nonic_ramp_function


def mass_source(psi, mass_psi):
    return mass_psi * mass_psi * psi


def vacuum_gb_scalar(
    weyl_electric_scalar,
    weyl_magnetic_scalar,
):
    return 8 * (weyl_electric_scalar - weyl_magnetic_scalar)


def vacuum_pontryagin_scalar(
    weyl_electric, weyl_magnetic, inverse_spatial_metric
):
    pontryagin_scalar = -16.0 * np.trace(
        weyl_electric
        @ inverse_spatial_metric
        @ weyl_magnetic
        @ inverse_spatial_metric
    )

    return pontryagin_scalar


def first_derivative_of_coupling_function(
    psi,
    coupling_parameters,
):
    linear_coupling = coupling_parameters[0]
    quadratic_coupling = coupling_parameters[1]
    quartic_coupling = coupling_parameters[2]

    df = 0.25 * (
        linear_coupling
        + quadratic_coupling * psi
        + quartic_coupling * np.power(psi, 3)
    )

    return df


def second_derivative_of_coupling_function(
    psi,
    coupling_parameters,
):
    quadratic_coupling = coupling_parameters[1]
    quartic_coupling = coupling_parameters[2]

    ddf = 0.25 * (
        quadratic_coupling + 3.0 * quartic_coupling * np.power(psi, 2)
    )

    return ddf


def negative_deriv_of_coupling_func(
    psi, coupling_parameters, start_time, ramp_time, time
):
    ramp_up_factor = nonic_ramp_function(
        time=time, t_start=start_time, t_ramp=ramp_time
    )
    ramped_up_coupling_parameters = ramp_up_factor * np.array(
        coupling_parameters
    )
    df = first_derivative_of_coupling_function(
        psi=psi,
        coupling_parameters=ramped_up_coupling_parameters,
    )
    result = -df
    return result


def negative_second_deriv_of_coupling_func(
    psi, coupling_parameters, start_time, ramp_time, time
):
    ramp_up_factor = nonic_ramp_function(
        time=time, t_start=start_time, t_ramp=ramp_time
    )
    ramped_up_coupling_parameters = ramp_up_factor * np.array(
        coupling_parameters
    )
    ddf = second_derivative_of_coupling_function(
        psi=psi,
        coupling_parameters=ramped_up_coupling_parameters,
    )
    result = -ddf
    return result


def gauss_bonnet_scalar_source(
    weyl_electric_scalar,
    weyl_magnetic_scalar,
    psi,
    coupling_parameters,
    mass_psi,
    t_start,
    t_ramp,
    time,
):
    ramp_up_factor = nonic_ramp_function(
        time=time, t_start=t_start, t_ramp=t_ramp
    )
    ramped_up_coupling_parameters = ramp_up_factor * np.array(
        coupling_parameters
    )
    scalar_source = vacuum_gb_scalar(
        weyl_electric_scalar=weyl_electric_scalar,
        weyl_magnetic_scalar=weyl_magnetic_scalar,
    )
    # The source function has a minus sign factor in the curvature coupling
    # function
    scalar_source *= -first_derivative_of_coupling_function(
        psi=psi, coupling_parameters=ramped_up_coupling_parameters
    )
    scalar_source += mass_source(psi=psi, mass_psi=mass_psi)

    return scalar_source


def add_scalar_source_to_dt_pi_scalar(scalar_source, lapse):
    return 0.1234 + lapse * scalar_source
