# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

_atmosphere_density_cutoff = 1.0e-8
_magnetic_field_magnitude_for_hydro = 1.0e-30
_polytropic_constant = 100.0
_polytropic_gamma = 2.0


def dg_package_data(
    tilde_d,
    tilde_ye,
    tilde_tau,
    tilde_s,
    tilde_b,
    tilde_phi,
    flux_tilde_d,
    flux_tilde_ye,
    flux_tilde_tau,
    flux_tilde_s,
    flux_tilde_b,
    flux_tilde_phi,
    lapse,
    shift,
    spatial_velocity_one_form,
    spatial_metric,
    rest_mass_density,
    electron_fraction,
    temperature,
    spatial_velocity,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    equation_of_state,
):
    def compute_char(lapse_sign):
        magnetic_field_magnitude = np.sqrt(np.einsum("i,i->", tilde_b, tilde_b))
        if (
            magnetic_field_magnitude < _magnetic_field_magnitude_for_hydro
            and rest_mass_density > _atmosphere_density_cutoff
        ):
            # Sound speeds
            pressure = (
                _polytropic_constant * rest_mass_density**_polytropic_gamma
            )
            sound_speed_squared = min(
                max(
                    (
                        _polytropic_gamma
                        * (_polytropic_gamma - 1.0)
                        * pressure
                        / (
                            rest_mass_density * (_polytropic_gamma - 1.0)
                            + _polytropic_gamma * pressure
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
            # As in the C++: v^2 is clamped, v.n is not.
            velocity_dot_normal = np.einsum(
                "i,i->", spatial_velocity, normal_covector
            )
            velocity_squared = min(
                max(
                    np.einsum(
                        "i,i->", spatial_velocity, spatial_velocity_one_form
                    ),
                    0.0,
                ),
                1.0 - 1.0e-8,
            )
            one_over_lorentz_factor_squared = 1.0 - velocity_squared
            d = np.sqrt(
                max(
                    0.0,
                    sound_speed_squared
                    * one_over_lorentz_factor_squared
                    * (
                        1.0
                        - velocity_squared * sound_speed_squared
                        - velocity_dot_normal**2 * (1.0 - sound_speed_squared)
                    ),
                )
            )
            Lambda = (
                lapse
                / (1.0 - velocity_squared * sound_speed_squared)
                * (
                    velocity_dot_normal * (1.0 - sound_speed_squared)
                    + lapse_sign * d
                )
            )
            return np.asarray(
                (Lambda - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    Lambda
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
                )
            )
        else:
            # Light speeds
            return np.asarray(
                (lapse_sign * lapse - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    lapse_sign * lapse
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
                )
            )

    def compute_fast(lapse_sign):
        # Without a magnetic field and in the atmosphere the fast speeds are
        # the largest speeds. Otherwise they are the estimate
        # a^2 = c_s^2 + v_A^2 (1 - c_s^2) through the dispersion relation of
        # the sound speeds, with this side's lapse, shift and spatial metric
        # and the inputs clamped as there.
        if (
            np.sqrt(np.einsum("i,i->", tilde_b, tilde_b))
            < _magnetic_field_magnitude_for_hydro
            or rest_mass_density <= _atmosphere_density_cutoff
        ):
            return compute_char(lapse_sign)
        polytropic_pressure = (
            _polytropic_constant * rest_mass_density**_polytropic_gamma
        )
        specific_internal_energy = polytropic_pressure / (
            rest_mass_density * (_polytropic_gamma - 1.0)
        )
        sound_speed_squared = min(
            max(
                _polytropic_gamma
                * (_polytropic_gamma - 1.0)
                * polytropic_pressure
                / (
                    rest_mass_density * (_polytropic_gamma - 1.0)
                    + _polytropic_gamma * polytropic_pressure
                ),
                0.0,
            ),
            1.0,
        )
        velocity_squared = min(
            max(np.dot(spatial_velocity, spatial_velocity_one_form), 0.0),
            1.0 - 1.0e-8,
        )
        velocity_dot_normal = np.dot(spatial_velocity, normal_covector)
        # B^i = tilde_b^i / sqrt(det(spatial_metric))
        comoving_magnetic_field_squared = (
            np.einsum("i,ij,j", tilde_b, spatial_metric, tilde_b)
            * (1.0 - velocity_squared)
            + np.dot(tilde_b, spatial_velocity_one_form) ** 2
        ) / np.linalg.det(spatial_metric)
        alfven_speed_squared = min(
            max(
                comoving_magnetic_field_squared
                / (
                    comoving_magnetic_field_squared
                    + rest_mass_density * (1.0 + specific_internal_energy)
                    + polytropic_pressure
                ),
                0.0,
            ),
            1.0,
        )
        a_squared = sound_speed_squared + alfven_speed_squared * (
            1.0 - sound_speed_squared
        )
        one_minus_v2_a2 = 1.0 - velocity_squared * a_squared
        discriminant = np.sqrt(
            max(
                a_squared
                * (1.0 - velocity_squared)
                * (
                    one_minus_v2_a2 - velocity_dot_normal**2 * (1.0 - a_squared)
                ),
                0.0,
            )
        )
        Lambda = lapse * (
            velocity_dot_normal * (1.0 - a_squared) + lapse_sign * discriminant
        ) / one_minus_v2_a2 - np.dot(shift, normal_covector)
        if normal_dot_mesh_velocity is not None:
            Lambda = Lambda - normal_dot_mesh_velocity
        return np.asarray(Lambda)

    # The return order matches dg_package_field_tags.
    return (
        tilde_d,
        tilde_ye,
        tilde_tau,
        tilde_s,
        tilde_b,
        tilde_phi,
        np.asarray(np.dot(flux_tilde_d, normal_covector)),
        np.asarray(np.dot(flux_tilde_ye, normal_covector)),
        np.asarray(np.dot(flux_tilde_tau, normal_covector)),
        np.einsum("ij,i->j", flux_tilde_s, normal_covector),
        np.einsum("ij,i->j", flux_tilde_b, normal_covector),
        np.asarray(np.dot(flux_tilde_phi, normal_covector)),
        compute_char(1.0),
        compute_char(-1.0),
        compute_fast(1.0),
        compute_fast(-1.0),
        normal_covector,
        # Raised with this side's spatial metric, not taken from normal_vector
        np.einsum("ij,j->i", np.linalg.inv(spatial_metric), normal_covector),
    )


def dg_boundary_terms(
    interior_tilde_d,
    interior_tilde_ye,
    interior_tilde_tau,
    interior_tilde_s,
    interior_tilde_b,
    interior_tilde_phi,
    interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_ye,
    interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    interior_fast_outgoing_char_speed,
    interior_fast_ingoing_char_speed,
    interior_interface_unit_normal,
    interior_interface_unit_normal_vector,
    exterior_tilde_d,
    exterior_tilde_ye,
    exterior_tilde_tau,
    exterior_tilde_s,
    exterior_tilde_b,
    exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_ye,
    exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    exterior_fast_outgoing_char_speed,
    exterior_fast_ingoing_char_speed,
    exterior_interface_unit_normal,
    exterior_interface_unit_normal_vector,
    use_strong_form,
):
    # Light-speed (divergence-cleaning) bounds: for Phi and the normal B.
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )
    # Fast-magnetosonic bounds for the MHD variables: the two sides' own fast
    # speeds, combined by the same min/max as the light bounds. The minus signs
    # on the exterior speeds are there because the neighbour packaged against
    # its own outward normal.
    fast_max = np.maximum(
        0.0,
        np.maximum(
            interior_fast_outgoing_char_speed,
            -exterior_fast_ingoing_char_speed,
        ),
    )
    fast_min = np.minimum(
        0.0,
        np.minimum(
            interior_fast_ingoing_char_speed,
            -exterior_fast_outgoing_char_speed,
        ),
    )

    def hll(l_max, l_min, u_int, nf_int, u_ext, nf_ext):
        l_int = l_min if use_strong_form else l_max
        return np.asarray(
            (l_int * nf_int + l_min * nf_ext + l_max * l_min * (u_ext - u_int))
            / (l_max - l_min)
        )

    tilde_d = hll(
        fast_max,
        fast_min,
        interior_tilde_d,
        interior_normal_dot_flux_tilde_d,
        exterior_tilde_d,
        exterior_normal_dot_flux_tilde_d,
    )
    tilde_ye = hll(
        fast_max,
        fast_min,
        interior_tilde_ye,
        interior_normal_dot_flux_tilde_ye,
        exterior_tilde_ye,
        exterior_normal_dot_flux_tilde_ye,
    )
    tilde_tau = hll(
        fast_max,
        fast_min,
        interior_tilde_tau,
        interior_normal_dot_flux_tilde_tau,
        exterior_tilde_tau,
        exterior_normal_dot_flux_tilde_tau,
    )
    tilde_phi = hll(
        lambda_max,
        lambda_min,
        interior_tilde_phi,
        interior_normal_dot_flux_tilde_phi,
        exterior_tilde_phi,
        exterior_normal_dot_flux_tilde_phi,
    )
    tilde_s = hll(
        fast_max,
        fast_min,
        interior_tilde_s,
        interior_normal_dot_flux_tilde_s,
        exterior_tilde_s,
        exterior_normal_dot_flux_tilde_s,
    )
    # Magnetic field: normal part (light) + tangential part (fast), split
    # along the average m of the two sides' unit normal covectors and vectors,
    # with n_i n^j = m_i m^j / (m_k m^k).
    n_lower = 0.5 * (
        interior_interface_unit_normal - exterior_interface_unit_normal
    )
    n_upper = 0.5 * (
        interior_interface_unit_normal_vector
        - exterior_interface_unit_normal_vector
    )
    n_upper = n_upper / np.dot(n_lower, n_upper)
    bn_int = np.dot(interior_tilde_b, n_lower)
    bn_ext = np.dot(exterior_tilde_b, n_lower)
    nfbn_int = np.dot(interior_normal_dot_flux_tilde_b, n_lower)
    nfbn_ext = np.dot(exterior_normal_dot_flux_tilde_b, n_lower)
    g_bn = hll(lambda_max, lambda_min, bn_int, nfbn_int, bn_ext, nfbn_ext)
    bt_int = interior_tilde_b - bn_int * n_upper
    bt_ext = exterior_tilde_b - bn_ext * n_upper
    nfbt_int = interior_normal_dot_flux_tilde_b - nfbn_int * n_upper
    nfbt_ext = exterior_normal_dot_flux_tilde_b - nfbn_ext * n_upper
    g_bt = hll(fast_max, fast_min, bt_int, nfbt_int, bt_ext, nfbt_ext)
    tilde_b = g_bn * n_upper + g_bt

    return (
        tilde_d,
        tilde_ye,
        tilde_tau,
        tilde_s,
        tilde_b,
        tilde_phi,
    )
