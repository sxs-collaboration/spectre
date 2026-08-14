# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

x_coords = [-5.0, 6.0]
momentum_left = [-0.01, -0.01, -0.01]
momentum_right = [0.01, 0.01, 0.01]
y_offset = 0.02
z_offset = 0.01
masses = [1.1, 0.43]
attenuation_parameter = 1.1
attenuation_radius = 3.6


def x_left(x):
    x_left = np.array(x)
    x_left[0] -= x_coords[0]
    x_left[1] -= y_offset
    x_left[2] -= z_offset

    velocity = np.array(momentum_left) / masses[0]
    beta2 = np.dot(velocity, velocity)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = gamma
    boost_matrix[0, 1:] = -gamma * velocity
    boost_matrix[1:, 0] = -gamma * velocity
    boost_matrix[1:, 1:] = (
        np.identity(3) + (gamma - 1.0) * np.outer(velocity, velocity) / beta2
    )

    boosted_vector = np.dot(boost_matrix, np.append(0.0, x_left))

    return boosted_vector[1:]


def x_right(x):
    x_right = np.array(x)
    x_right[0] -= x_coords[1]
    x_right[1] -= y_offset
    x_right[2] -= z_offset

    velocity = np.array(momentum_right) / masses[1]
    beta2 = np.dot(velocity, velocity)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = gamma
    boost_matrix[0, 1:] = -gamma * velocity
    boost_matrix[1:, 0] = -gamma * velocity
    boost_matrix[1:, 1:] = (
        np.identity(3) + (gamma - 1.0) * np.outer(velocity, velocity) / beta2
    )

    boosted_vector = np.dot(boost_matrix, np.append(0.0, x_right))

    return boosted_vector[1:]


def spacetime_left(x):
    r = np.linalg.norm(x_left(x))
    conformalfactor = 1.0 + 0.5 * masses[0] / r
    lapse = (1.0 - 0.5 * masses[0] / r) / conformalfactor
    shift = np.zeros(3)
    spatial_metric = np.identity(3) * pow(conformalfactor, 4)

    spacetime_metric = np.zeros((4, 4))
    spacetime_metric[0, 0] = -(lapse**2) + np.dot(
        shift, np.dot(spatial_metric, shift)
    )
    spacetime_metric[0, 1:] = np.dot(spatial_metric, shift)
    spacetime_metric[1:, 0] = spacetime_metric[0, 1:]
    spacetime_metric[1:, 1:] = spatial_metric

    return spacetime_metric


def spacetime_right(x):
    r = np.linalg.norm(x_right(x))
    conformalfactor = 1.0 + 0.5 * masses[1] / r
    lapse = (1.0 - 0.5 * masses[1] / r) / conformalfactor
    shift = np.zeros(3)
    spatial_metric = np.identity(3) * pow(conformalfactor, 4)

    spacetime_metric = np.zeros((4, 4))
    spacetime_metric[0, 0] = -(lapse**2) + np.dot(
        shift, np.dot(spatial_metric, shift)
    )
    spacetime_metric[0, 1:] = np.dot(spatial_metric, shift)
    spacetime_metric[1:, 0] = spacetime_metric[0, 1:]
    spacetime_metric[1:, 1:] = spatial_metric

    return spacetime_metric


def boost_spacetime_metric(spacetime_metric, velocity):
    beta2 = np.dot(velocity, velocity)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = gamma
    boost_matrix[0, 1:] = -gamma * velocity
    boost_matrix[1:, 0] = -gamma * velocity
    boost_matrix[1:, 1:] = (
        np.identity(3) + (gamma - 1.0) * np.outer(velocity, velocity) / beta2
    )

    return np.dot(boost_matrix, np.dot(spacetime_metric, boost_matrix.T))


def shift(spacetime_metric):
    spatial_metric = spacetime_metric[1:, 1:]
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    g_jt = spacetime_metric[1:, 0]

    shift = np.dot(inverse_spatial_metric, g_jt)

    return shift


def lapse(spacetime_metric):
    spatial_metric = spacetime_metric[1:, 1:]
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    g_jt = spacetime_metric[1:, 0]
    shift = np.dot(inverse_spatial_metric, g_jt)
    beta_g_it = np.dot(shift, g_jt)
    g_tt = spacetime_metric[0, 0]

    lapse = np.sqrt(beta_g_it - g_tt)

    return lapse


def superposed_spacetime_metric(x):
    spacetime_metric_left = spacetime_left(x)
    spacetime_metric_right = spacetime_right(x)
    boosted_spacetime_metric_left = boost_spacetime_metric(
        spacetime_metric_left, np.array(momentum_left) / masses[0]
    )
    boosted_spacetime_metric_right = boost_spacetime_metric(
        spacetime_metric_right, np.array(momentum_right) / masses[1]
    )

    spatial_metric_left = boosted_spacetime_metric_left[1:, 1:]
    spatial_metric_right = boosted_spacetime_metric_right[1:, 1:]
    shift_left = shift(boosted_spacetime_metric_left)
    shift_right = shift(boosted_spacetime_metric_right)
    lapse_left = lapse(boosted_spacetime_metric_left)
    lapse_right = lapse(boosted_spacetime_metric_right)

    superposed_spatial_metric = (
        spatial_metric_left + spatial_metric_right - np.identity(3)
    )
    superposed_shift = shift_left + shift_right
    superposed_lapse = lapse_left * lapse_right

    superposed_spacetime_metric = np.zeros((4, 4))
    superposed_spacetime_metric[0, 0] = -(superposed_lapse**2) + np.dot(
        superposed_shift, np.dot(superposed_spatial_metric, superposed_shift)
    )
    superposed_spacetime_metric[0, 1:] = np.dot(
        superposed_spatial_metric, superposed_shift
    )
    superposed_spacetime_metric[1:, 0] = superposed_spacetime_metric[0, 1:]
    superposed_spacetime_metric[1:, 1:] = superposed_spatial_metric

    return superposed_spacetime_metric


def conformal_metric_bbh_isotropic(x):
    return superposed_spacetime_metric(x)[1:, 1:]


def inv_conformal_metric_bbh_isotropic(x):
    return np.linalg.inv(conformal_metric_bbh_isotropic(x))


def shift_background(x):
    return np.zeros(3)


def longitudinal_shift_background_bbh_isotropic(x):
    return np.zeros((3, 3))


def conformal_factor_minus_one_bbh_isotropic(x):
    return 0.0


def energy_density_bbh_isotropic(x):
    return 0.0


def stress_trace_bbh_isotropic(x):
    return 0.0


def momentum_density_bbh_isotropic(x):
    return np.zeros(3)
