# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

xcoords = [-4.5, 10.2]
mass_left = 1.1
mass_right = 0.9
atenuation_parameter = 0.99
outer_radius = 21.0
separation = xcoords[1] - xcoords[0]
total_mass = mass_left + mass_right
reduced_mass = mass_left * mass_right / total_mass
p_circular_squared = (
    reduced_mass * reduced_mass * total_mass / separation
    + 4.0
    * reduced_mass
    * reduced_mass
    * total_mass
    * total_mass
    / (separation * separation)
    + (74.0 - 43.0 * reduced_mass / total_mass)
    * reduced_mass
    * reduced_mass
    * total_mass
    * total_mass
    * total_mass
    / (8.0 * separation * separation * separation)
)
momentum1 = np.array([0.0, np.sqrt(p_circular_squared), 0.0])
momentum2 = np.array([0.0, -np.sqrt(p_circular_squared), 0.0])
delta = np.identity(3)
normal = np.array([-1.0, 0.0, 0.0])


def distance_left(x):
    return np.sqrt(
        (x[0] - xcoords[0]) * (x[0] - xcoords[0]) + x[1] * x[1] + x[2] * x[2]
    )


def distance_right(x):
    return np.sqrt(
        (x[0] - xcoords[1]) * (x[0] - xcoords[1]) + x[1] * x[1] + x[2] * x[2]
    )


def deriv_one_over_distance_left(x):
    return -(x - np.array([xcoords[0], 0.0, 0.0])) / (
        distance_left(x) * distance_left(x) * distance_left(x)
    )


def deriv_one_over_distance_right(x):
    return -(x - np.array([xcoords[1], 0.0, 0.0])) / (
        distance_right(x) * distance_right(x) * distance_right(x)
    )


def deriv_3_distance_left(x):
    deriv_3_distance_left_aux = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                deriv_3_distance_left_aux[i, j, k] = (
                    -normal_left(x)[k] * delta[i][j]
                    - normal_left(x)[i] * delta[j][k]
                    - normal_left(x)[j] * delta[i][k]
                    + 3
                    * normal_left(x)[i]
                    * normal_left(x)[j]
                    * normal_left(x)[k]
                ) / np.square(distance_left(x))
    return deriv_3_distance_left_aux


def deriv_3_distance_right(x):
    deriv_3_distance_right_aux = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                deriv_3_distance_right_aux[i, j, k] = (
                    -normal_right(x)[k] * delta[i][j]
                    - normal_right(x)[i] * delta[j][k]
                    - normal_right(x)[j] * delta[i][k]
                    + 3
                    * normal_right(x)[i]
                    * normal_right(x)[j]
                    * normal_right(x)[k]
                ) / np.square(distance_right(x))
    return deriv_3_distance_right_aux


def normal_left(x):
    return (x - np.array([xcoords[0], 0.0, 0.0])) / distance_left(x)


def normal_right(x):
    return (x - np.array([xcoords[1], 0.0, 0.0])) / distance_right(x)


def near_zone_term(x):
    r_left = distance_left(x)
    r_right = distance_right(x)
    n_left = normal_left(x)
    n_right = normal_right(x)
    s = r_left + r_right + separation
    near_zone_term_aux = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            near_zone_term_aux[i, j] += (
                0.25
                / (mass_left * r_left)
                * (
                    (
                        np.dot(momentum1, momentum1)
                        - 5.0
                        * np.dot(n_left, momentum1)
                        * np.dot(n_left, momentum1)
                    )
                    * delta[i][j]
                    + 2.0 * momentum1[i] * momentum1[j]
                    + (
                        3.0
                        * np.dot(n_left, momentum1)
                        * np.dot(n_left, momentum1)
                        - 5.0 * np.dot(momentum1, momentum1)
                    )
                    * n_left[i]
                    * n_left[j]
                    + 6.0
                    * np.dot(n_left, momentum1)
                    * (n_left[i] * momentum1[j] + n_left[j] * momentum1[i])
                )
                + 0.25
                / (mass_right * r_right)
                * (
                    (
                        np.dot(momentum2, momentum2)
                        - 5.0
                        * np.dot(n_right, momentum2)
                        * np.dot(n_right, momentum2)
                    )
                    * delta[i][j]
                    + 2.0 * momentum2[i] * momentum2[j]
                    + (
                        3.0
                        * np.dot(n_right, momentum2)
                        * np.dot(n_right, momentum2)
                        - 5.0 * np.dot(momentum2, momentum2)
                    )
                    * n_right[i]
                    * n_right[j]
                    + 6.0
                    * np.dot(n_right, momentum2)
                    * (n_right[i] * momentum2[j] + n_right[j] * momentum2[i])
                )
                + 0.125
                * (mass_left * mass_right)
                * (
                    -32.0
                    / s
                    * (1.0 / separation + 1.0 / s)
                    * normal[i]
                    * normal[j]
                    + 2.0
                    * (
                        (r_left + r_right)
                        / (separation * separation * separation)
                        + 12.0 / (s * s)
                    )
                    * n_left[i]
                    * n_right[j]
                    + 16.0
                    * (2.0 / (s * s) - 1.0 / (separation * separation))
                    * (n_left[i] * normal[j] + n_left[j] * normal[i])
                    + (
                        5.0 / (separation * r_left)
                        - 1
                        / (separation * separation * separation)
                        * (r_right * r_right / r_left + 3.0 * r_left)
                        - 8.0 / s * (1.0 / r_left + 1.0 / s)
                    )
                    * n_left[i]
                    * n_left[j]
                    + (
                        5.0
                        * r_left
                        / (separation * separation * separation)
                        * (r_left / r_right - 1.0)
                        - 17.0 / (separation * r_left)
                        + 4.0 / (r_left * r_right)
                        + 8.0 / s * (1.0 / r_left + 4.0 / separation)
                    )
                    * delta[i][j]
                )
                + 0.125
                * (mass_left * mass_right)
                * (
                    -32.0
                    / s
                    * (1.0 / separation + 1.0 / s)
                    * normal[i]
                    * normal[j]
                    + 2.0
                    * (
                        (r_left + r_right)
                        / (separation * separation * separation)
                        + 12.0 / (s * s)
                    )
                    * n_right[i]
                    * n_left[j]
                    - 16.0
                    * (2.0 / (s * s) - 1 / (separation * separation))
                    * (n_right[i] * normal[j] + n_right[j] * normal[i])
                    + (
                        5.0 / (separation * r_right)
                        - 1.0
                        / (separation * separation * separation)
                        * (r_left * r_left / r_right + 3.0 * r_right)
                        - 8.0 / s * (1.0 / r_right + 1.0 / s)
                    )
                    * n_right[i]
                    * n_right[j]
                    + (
                        5.0
                        * r_right
                        / (separation * separation * separation)
                        * (r_right / r_left - 1.0)
                        - 17.0 / (separation * r_right)
                        + 4.0 / (r_left * r_right)
                        + 8.0 / s * (1.0 / r_right + 4.0 / separation)
                    )
                    * delta[i][j]
                )
            )
    return near_zone_term_aux


def present_term(x):
    r_left = distance_left(x)
    r_right = distance_right(x)
    n_left = normal_left(x)
    n_right = normal_right(x)
    s = r_left + r_right + separation
    present_term_aux = np.zeros((3, 3))
    u1_1 = np.array([0.0, 0.0, 0.0])
    u1_2 = np.array([0.0, 0.0, 0.0])
    u2 = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        u1_1[i] = momentum1[i] / np.sqrt(mass_left)
        u1_2[i] = momentum2[i] / np.sqrt(mass_right)
        u2[i] = np.sqrt(mass_left * mass_right / (2.0 * separation)) * normal[i]
    for i in range(3):
        for j in range(3):
            present_term_aux[i, j] += (
                -1.0
                / (4.0 * r_left)
                * (
                    (
                        np.dot(u1_1, u1_1)
                        - 5.0 * np.dot(u1_1, n_left) * np.dot(u1_1, n_left)
                    )
                    * delta[i][j]
                    + 2.0 * u1_1[i] * u1_1[j]
                    + (
                        3.0 * np.dot(u1_1, n_left) * np.dot(u1_1, n_left)
                        - 5.0 * np.dot(u1_1, u1_1)
                    )
                    * n_left[i]
                    * n_left[j]
                    + 6.0
                    * np.dot(u1_1, n_left)
                    * (n_left[i] * u1_1[j] + n_left[j] * u1_1[i])
                )
                - 1.0
                / (4.0 * r_right)
                * (
                    (
                        np.dot(u1_2, u1_2)
                        - 5 * np.dot(u1_2, n_right) * np.dot(u1_2, n_right)
                    )
                    * delta[i][j]
                    + 2.0 * u1_2[i] * u1_2[j]
                    + (
                        3.0 * np.dot(u1_2, n_right) * np.dot(u1_2, n_right)
                        - 5.0 * np.dot(u1_2, u1_2)
                    )
                    * n_right[i]
                    * n_right[j]
                    + 6.0
                    * np.dot(u1_2, n_right)
                    * (n_right[i] * u1_2[j] + n_right[j] * u1_2[i])
                )
                - 1.0
                / (4.0 * r_left)
                * (
                    (
                        np.dot(u2, u2)
                        - 5.0 * np.dot(u2, n_left) * np.dot(u2, n_left)
                    )
                    * delta[i][j]
                    + 2.0 * u2[i] * u2[j]
                    + (
                        3.0 * np.dot(u2, n_left) * np.dot(u2, n_left)
                        - 5.0 * np.dot(u2, u2)
                    )
                    * n_left[i]
                    * n_left[j]
                    + 6.0
                    * np.dot(u2, n_left)
                    * (n_left[i] * u2[j] + n_left[j] * u2[i])
                )
                - 1.0
                / (4.0 * r_right)
                * (
                    (
                        np.dot(u2, u2)
                        - 5.0 * np.dot(u2, n_right) * np.dot(u2, n_right)
                    )
                    * delta[i][j]
                    + 2.0 * u2[i] * u2[j]
                    + (
                        3.0 * np.dot(u2, n_right) * np.dot(u2, n_right)
                        - 5.0 * np.dot(u2, u2)
                    )
                    * n_right[i]
                    * n_right[j]
                    + 6.0
                    * np.dot(u2, n_right)
                    * (n_right[i] * u2[j] + n_right[j] * u2[i])
                )
            )
    return present_term_aux


def radiative_term(x):
    return (
        np.zeros((3, 3)) + near_zone_term(x) + present_term(x)
    )  # + past_term(x) + integral_term(x)


def pn_conjugate_momentum3(x):
    pn_conjugate_momentum3_aux = np.zeros((3, 3))
    deriv_frac_r_left = deriv_one_over_distance_left(x)
    deriv_frac_r_right = deriv_one_over_distance_right(x)
    deriv_3_r_left = deriv_3_distance_left(x)
    deriv_3_r_right = deriv_3_distance_right(x)
    for i in range(3):
        for j in range(3):
            sum = 0.0
            for k in range(3):
                term1_1 = -delta[i, j] * deriv_frac_r_left[k]
                term2_1 = 2 * (
                    delta[i, k] * deriv_frac_r_left[j]
                    + delta[j, k] * deriv_frac_r_left[i]
                )
                term3_1 = -0.5 * deriv_3_r_left[i, j, k]
                term1_2 = -delta[i, j] * deriv_frac_r_right[k]
                term2_2 = 2 * (
                    delta[i, k] * deriv_frac_r_right[j]
                    + delta[j, k] * deriv_frac_r_right[i]
                )
                term3_2 = -0.5 * deriv_3_r_right[i, j, k]
                sum += momentum1[k] * (term1_1 + term2_1 + term3_1) + momentum2[
                    k
                ] * (term1_2 + term2_2 + term3_2)
            pn_conjugate_momentum3_aux[i, j] = sum
    return pn_conjugate_momentum3_aux


def pn_extrinsic_curvature(x):
    r_left = distance_left(x)
    r_right = distance_right(x)
    E_left = (
        mass_left
        + np.dot(momentum1, momentum1) / (2.0 * mass_left)
        - mass_left * mass_right / (2.0 * separation)
    )
    E_right = (
        mass_right
        + np.dot(momentum2, momentum2) / (2.0 * mass_right)
        - mass_left * mass_right / (2.0 * separation)
    )
    conformal_factor = 1.0 + E_left / (2.0 * r_left) + E_right / (2.0 * r_right)
    tilde_pi = pn_conjugate_momentum3(x)
    return -1.0 / pow(conformal_factor, 10) * tilde_pi


def conformal_metric(x):
    r_left = distance_left(x)
    r_right = distance_right(x)
    E_left = (
        mass_left
        + np.dot(momentum1, momentum1) / (2.0 * mass_left)
        - mass_left * mass_right / (2.0 * separation)
    )
    E_right = (
        mass_right
        + np.dot(momentum2, momentum2) / (2.0 * mass_right)
        - mass_left * mass_right / (2.0 * separation)
    )
    conformal_factor = 1.0 + E_left / (2.0 * r_left) + E_right / (2.0 * r_right)
    pn_conformal_term = np.diag(np.full(3, np.power(conformal_factor, 4)))
    att_func = 1.0 / (
        (
            1.0
            + atenuation_parameter
            * atenuation_parameter
            * mass_left
            * mass_left
            / (r_left * r_left)
        )
        * (
            1.0
            + atenuation_parameter
            * atenuation_parameter
            * mass_right
            * mass_right
            / (r_right * r_right)
        )
    )
    return pn_conformal_term + att_func * radiative_term(x)


def deriv_conformal_metric(x):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace(x):
    inv_conformal_metric = np.linalg.inv(conformal_metric(x))
    extrinsic_curvature = pn_extrinsic_curvature(x)
    extrinsic_curvature_trace_aux = 0.0
    for i in range(3):
        for j in range(3):
            extrinsic_curvature_trace_aux += (
                inv_conformal_metric[i, j] * extrinsic_curvature[i, j]
            )
    return extrinsic_curvature_trace_aux


def shift_background(x):
    return np.zeros(3)


def longitudinal_shift_background(x):
    return np.zeros((3, 3))


def conformal_factor_minus_one(x):
    return 0.0


def energy_density(x):
    return 0.0


def stress_trace(x):
    return 0.0


def momentum_density(x):
    return np.zeros(3)
