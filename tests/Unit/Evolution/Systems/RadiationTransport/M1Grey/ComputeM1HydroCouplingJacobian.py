# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def neutrino_source_jacobian(
    tilde_s,
    tilde_e,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    fluid_velocity,
    fluid_lorentz_factor,
    closure_factor,
    comoving_energy_density,
    comoving_momentum_density_spatial,
    comoving_momentum_density_normal,
    lapse,
    spatial_metric,
    inverse_spatial_metric,
):
    s_squared_floor = 1.0e-150

    d_thin = (
        0.2
        * closure_factor**2
        * (3.0 + closure_factor * (-1.0 + 3.0 * closure_factor))
    )

    d_thick = 1.0 - d_thin

    fluid_velocity_lower = np.einsum("a, ia", fluid_velocity, spatial_metric)

    s_upper = np.einsum("a, ia", tilde_s, inverse_spatial_metric)

    # np.einsum("a, ia", , )

    comoving_four_momentum_density_upper = (
        np.einsum(
            "a, ia", comoving_momentum_density_spatial, inverse_spatial_metric
        )
        + comoving_momentum_density_normal * fluid_velocity
    )

    inverse_s_norm = 1.0 / (
        np.einsum("a, a", s_upper, tilde_s) + s_squared_floor
    )

    fluid_velocity_norm = np.einsum(
        "a, a", fluid_velocity, fluid_velocity_lower
    )

    s_dot_fluid_velocity = np.einsum("a, a", tilde_s, fluid_velocity)

    s_dot_fluid_velocity_normalized = s_dot_fluid_velocity * inverse_s_norm

    s_dot_fluid_velocity_squared_normalized = (
        s_dot_fluid_velocity * s_dot_fluid_velocity_normalized
    )

    denom = 1.0 / (1.0 + 2.0 * fluid_lorentz_factor**2)

    scaled_comoving_energy_density = (
        closure_factor**2 * comoving_energy_density
    )

    h_difference_s_coef = fluid_lorentz_factor * (
        fluid_velocity_norm - tilde_e * s_dot_fluid_velocity_normalized
    )

    common_difference_term = denom * (
        (2.0 * fluid_lorentz_factor**2 - 3.0) * tilde_e
        - 4.0 * fluid_lorentz_factor**2 * s_dot_fluid_velocity
    )

    j_difference = fluid_lorentz_factor**2 * (
        fluid_velocity_norm * common_difference_term
        + tilde_e * s_dot_fluid_velocity_squared_normalized
    )

    h_difference_v_coef = fluid_lorentz_factor * (
        common_difference_term + j_difference + s_dot_fluid_velocity
    )

    h_difference = (
        h_difference_s_coef * tilde_s
        - h_difference_v_coef * fluid_velocity_lower
    )

    deriv_e_h_v_coef = fluid_lorentz_factor**2 * (
        -4.0 * d_thick * denom
        - d_thin * (1.0 + s_dot_fluid_velocity_squared_normalized)
    )

    deriv_e_h_s_coef = d_thin * s_dot_fluid_velocity_normalized

    deriv_s_h_trace_coef = (
        1.0 / fluid_lorentz_factor + d_thin * h_difference_s_coef
    ) / fluid_lorentz_factor

    deriv_s_j_v_coef = (
        -2.0
        * fluid_lorentz_factor
        * (deriv_s_h_trace_coef + d_thick * fluid_velocity_norm * denom)
    )

    deriv_s_h_vv_coef = (
        2.0
        - denom * d_thick
        + 2.0 * d_thin * fluid_lorentz_factor * h_difference_s_coef
    )

    deriv_s_h_vs_coef = d_thin * tilde_e * inverse_s_norm

    deriv_s_h_ss_coef = (
        2.0 * s_dot_fluid_velocity_normalized * deriv_s_h_vs_coef
    )

    deriv_s_j_s_coef = (
        -2.0
        * fluid_lorentz_factor
        * s_dot_fluid_velocity_squared_normalized
        * deriv_s_h_vs_coef
    )

    deriv_s_h_sv_coef = fluid_lorentz_factor * deriv_s_j_s_coef

    constant_d_deriv_e_j_over_lorentz_factor = fluid_lorentz_factor * (
        d_thin * (1.0 + s_dot_fluid_velocity_squared_normalized)
        + 3.0 * d_thick * denom * (1.0 + fluid_velocity_norm)
    )

    constant_d_deriv_s_j_over_lorentz_factor = (
        deriv_s_j_v_coef * fluid_velocity + deriv_s_j_s_coef * s_upper
    )

    constant_d_deriv_e_h_over_lorentz_factor = (
        deriv_e_h_v_coef * fluid_velocity_lower - deriv_e_h_s_coef * tilde_s
    )

    # needs outer product
    constant_d_deriv_s_h_over_lorentz_factor = (
        deriv_s_h_vv_coef * np.outer(fluid_velocity, fluid_velocity_lower)
        + deriv_s_h_ss_coef * np.outer(s_upper, tilde_s)
        - deriv_s_h_sv_coef * np.outer(s_upper, fluid_velocity_lower)
        - deriv_s_h_vs_coef * np.outer(fluid_velocity, tilde_s)
    )

    loop_bound = 3
    for i in range(0, loop_bound):
        constant_d_deriv_s_h_over_lorentz_factor[i, i] += deriv_s_h_trace_coef

    deriv_dthin_prefactor = 1.0 / (
        scaled_comoving_energy_density * j_difference
        - np.einsum("a, a", comoving_four_momentum_density_upper, h_difference)
        + 5.0
        / 3.0
        * comoving_energy_density**2
        / (2.0 + closure_factor * (-1.0 + closure_factor * 4.0))
    )

    deriv_e_dthin_over_lorentz_factor = deriv_dthin_prefactor * (
        np.einsum(
            "a, a",
            comoving_four_momentum_density_upper,
            constant_d_deriv_e_h_over_lorentz_factor,
        )
        - scaled_comoving_energy_density
        * constant_d_deriv_e_j_over_lorentz_factor
    )

    deriv_s_dthin_over_lorentz_factor = deriv_dthin_prefactor * (
        np.einsum(
            "a, ia",
            comoving_four_momentum_density_upper,
            constant_d_deriv_s_h_over_lorentz_factor,
        )
        - scaled_comoving_energy_density
        * constant_d_deriv_s_j_over_lorentz_factor
    )

    deriv_e_j = fluid_lorentz_factor * (
        constant_d_deriv_e_j_over_lorentz_factor
        + j_difference * deriv_e_dthin_over_lorentz_factor
    )

    deriv_s_j = fluid_lorentz_factor * (
        constant_d_deriv_s_j_over_lorentz_factor
        + j_difference * deriv_s_dthin_over_lorentz_factor
    )

    deriv_e_h = fluid_lorentz_factor * (
        constant_d_deriv_e_h_over_lorentz_factor
        + deriv_e_dthin_over_lorentz_factor * h_difference
    )

    deriv_s_h = fluid_lorentz_factor * (
        constant_d_deriv_s_h_over_lorentz_factor
        + np.outer(deriv_s_dthin_over_lorentz_factor, h_difference)
    )

    deriv_source_e_j_coef = lapse * fluid_lorentz_factor * scattering_opacity

    deriv_source_e_v_coef = (
        lapse * fluid_lorentz_factor * (absorption_opacity + scattering_opacity)
    )

    deriv_source_s_h_coef = -lapse * (absorption_opacity + scattering_opacity)

    deriv_source_s_jv_coef = -lapse * fluid_lorentz_factor * absorption_opacity

    # returned source terms
    deriv_e_source_e = deriv_source_e_j_coef * deriv_e_j - deriv_source_e_v_coef

    deriv_s_source_e = (
        deriv_source_e_j_coef * deriv_s_j
        + deriv_source_e_v_coef * fluid_velocity
    )

    deriv_e_source_s = (
        deriv_source_s_h_coef * deriv_e_h
        + deriv_source_s_jv_coef * deriv_e_j * fluid_velocity_lower
    )

    deriv_s_source_s = (
        deriv_source_s_h_coef * deriv_s_h
        + deriv_source_s_jv_coef * np.outer(deriv_s_j, fluid_velocity_lower)
    )

    return (
        deriv_e_source_e,
        deriv_s_source_e,
        deriv_e_source_s,
        deriv_s_source_s,
    )


def hydro_coupling_jacobian_deriv_e_source_e(
    tilde_s,
    tilde_e,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    fluid_velocity,
    fluid_lorentz_factor,
    closure_factor,
    comoving_energy_density,
    comoving_momentum_density_spatial,
    comoving_momentum_density_normal,
    lapse,
    spatial_metric,
    inverse_spatial_metric,
):
    deriv_e_source_e, _, _, _ = neutrino_source_jacobian(
        tilde_s,
        tilde_e,
        emissivity,
        absorption_opacity,
        scattering_opacity,
        fluid_velocity,
        fluid_lorentz_factor,
        closure_factor,
        comoving_energy_density,
        comoving_momentum_density_spatial,
        comoving_momentum_density_normal,
        lapse,
        spatial_metric,
        inverse_spatial_metric,
    )

    result = deriv_e_source_e

    return result


def hydro_coupling_jacobian_deriv_e_source_s(
    tilde_s,
    tilde_e,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    fluid_velocity,
    fluid_lorentz_factor,
    closure_factor,
    comoving_energy_density,
    comoving_momentum_density_spatial,
    comoving_momentum_density_normal,
    lapse,
    spatial_metric,
    inverse_spatial_metric,
):
    _, _, deriv_e_source_s, _ = neutrino_source_jacobian(
        tilde_s,
        tilde_e,
        emissivity,
        absorption_opacity,
        scattering_opacity,
        fluid_velocity,
        fluid_lorentz_factor,
        closure_factor,
        comoving_energy_density,
        comoving_momentum_density_spatial,
        comoving_momentum_density_normal,
        lapse,
        spatial_metric,
        inverse_spatial_metric,
    )

    result = deriv_e_source_s

    return result


def hydro_coupling_jacobian_deriv_s_source_e(
    tilde_s,
    tilde_e,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    fluid_velocity,
    fluid_lorentz_factor,
    closure_factor,
    comoving_energy_density,
    comoving_momentum_density_spatial,
    comoving_momentum_density_normal,
    lapse,
    spatial_metric,
    inverse_spatial_metric,
):
    _, deriv_s_source_e, _, _ = neutrino_source_jacobian(
        tilde_s,
        tilde_e,
        emissivity,
        absorption_opacity,
        scattering_opacity,
        fluid_velocity,
        fluid_lorentz_factor,
        closure_factor,
        comoving_energy_density,
        comoving_momentum_density_spatial,
        comoving_momentum_density_normal,
        lapse,
        spatial_metric,
        inverse_spatial_metric,
    )

    result = deriv_s_source_e

    return result


def hydro_coupling_jacobian_deriv_s_source_s(
    tilde_s,
    tilde_e,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    fluid_velocity,
    fluid_lorentz_factor,
    closure_factor,
    comoving_energy_density,
    comoving_momentum_density_spatial,
    comoving_momentum_density_normal,
    lapse,
    spatial_metric,
    inverse_spatial_metric,
):
    _, _, _, deriv_s_source_s = neutrino_source_jacobian(
        tilde_s,
        tilde_e,
        emissivity,
        absorption_opacity,
        scattering_opacity,
        fluid_velocity,
        fluid_lorentz_factor,
        closure_factor,
        comoving_energy_density,
        comoving_momentum_density_spatial,
        comoving_momentum_density_normal,
        lapse,
        spatial_metric,
        inverse_spatial_metric,
    )

    result = deriv_s_source_s

    return result


# End of functions for testing M1HydroCoupling.cpp (Jacobian terms)
