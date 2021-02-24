# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# variable names chosen to visually correspond to the rendered mathematics.
# t = time
# r = extraction_radius
# a = amplitude
# k = duration
def pulse_profile_coefficient_a(t, r, a, k):
    u = t - r
    return 3.0 * a * np.exp(-u**2 / k**2) / (k**4 * r**5) * (
        3.0 * k**4 + 4.0 * r**2 * u**2 - 2.0 * k**2 * r * (r + 3.0 * u))


def pulse_profile_coefficient_b(t, r, a, k):
    u = t - r
    return 2.0 * a * np.exp(-u**2 / k**2) / (k**6 * r**5) * (
        -3.0 * k**6 + 4.0 * r**3 * u**3 - 6.0 * k**2 * r**2 * u *
        (r + u) + 3.0 * k**4 * r * (r + 2.0 * u))


def pulse_profile_coefficient_c(t, r, a, k):
    u = t - r
    return 0.25 * a * np.exp(-u**2 / k**2) / (k**8 * r**5) * (
        21.0 * k**8 + 16.0 * r**4 * u**4 - 16.0 * k**2 * r**3 * u**2 *
        (3.0 * r + u) - 6.0 * k**6 * r *
        (3.0 * r + 7.0 * u) + 12.0 * k**4 * r**2 *
        (r**2 + 2.0 * r * u + 3.0 * u**2))


def dt_pulse_profile_coefficient_a(t, r, a, k):
    u = t - r
    return (-6.0 * a * np.exp(-u**2 / k**2) / (k**6 * r**5) *
            (4.0 * r**2 * u**3 + 3.0 * k**4 * (r + u) - 6.0 * k**2 * r * u *
             (r + u)))


def dt_pulse_profile_coefficient_b(t, r, a, k):
    u = t - r
    return (4.0 * a * np.exp(-u**2 / k**2) / (k**8 * r**5) *
            (-4.0 * r**3 * u**4 + 3.0 * k**6 *
             (r + u) + 6.0 * k**2 * r**2 * u**2 *
             (2 * r + u) - 3.0 * k**4 * r * (r**2 + 3 * r * u + 2 * u**2)))


def dt_pulse_profile_coefficient_c(t, r, a, k):
    u = t - r
    return (-0.5 * a * np.exp(-u**2 / k**2) / (k**10 * r**5) *
            (16.0 * r**4 * u**5 + 21.0 * k**8 *
             (r + u) - 16.0 * k**2 * r**3 * u**3 *
             (5.0 * r + u) + 12.0 * k**4 * r**2 * u *
             (5.0 * r**2 + 4 * r * u + 3 * u**2) - 6.0 * k**6 * r *
             (2.0 * r**2 + 9.0 * r * u + 7.0 * u**2)))


def dr_pulse_profile_coefficient_a(t, r, a, k):
    u = t - r
    return -dt_pulse_profile_coefficient_a(t, r, a, k) + 3.0 * a * np.exp(
        -u**2 / k**2) / (k**4 * r**6) * (-15.0 * k**4 - 12.0 * r**2 * u**2 +
                                         6.0 * k**2 * r * (r + 4.0 * u))


def dr_pulse_profile_coefficient_b(t, r, a, k):
    u = t - r
    return -dt_pulse_profile_coefficient_b(t, r, a, k) + 2.0 * a * np.exp(
        -u**2 / k**2) / (k**6 * r**6) * (15.0 * k**6 - 8.0 * r**3 * u**3 +
                                         6.0 * k**2 * r**2 * u *
                                         (2.0 * r + 3.0 * u) - 3.0 * k**4 * r *
                                         (3.0 * r + 8.0 * u))


def dr_pulse_profile_coefficient_c(t, r, a, k):
    u = t - r
    return -dt_pulse_profile_coefficient_c(t, r, a, k) + 0.25 * a * np.exp(
        -u**2 / k**2) / (k**8 * r**6) * (
            -105.0 * k**8 - 16.0 * r**4 * u**4 + 16.0 * k**2 * r**3 * u**2 *
            (3.0 * r + 2.0 * u) + 6.0 * k**6 * r *
            (9.0 * r + 28.0 * u) - 12.0 * k**4 * r**2 *
            (r**2 + 4.0 * r * u + 9.0 * u**2))


def spherical_metric(sin_theta, cos_theta, t, r, a, k):
    f_r_r = 2.0 - 3.0 * sin_theta**2
    f_r_th = -3.0 * sin_theta * cos_theta
    fC_th_th = 3.0 * sin_theta**2
    fA_th_th = -1.0
    fC_ph_ph = -3.0 * sin_theta**2
    fA_ph_ph = 3.0 * sin_theta**2 - 1.0
    A = pulse_profile_coefficient_a(t, r, a, k)
    B = pulse_profile_coefficient_b(t, r, a, k)
    C = pulse_profile_coefficient_c(t, r, a, k)
    return np.array([
        [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0 + f_r_r * A, B * f_r_th * r, 0.0],
        [0.0, B * f_r_th * r, (1.0 + C * fC_th_th + A * fA_th_th) * r**2, 0.0],
        [0.0, 0.0, 0.0, (1.0 + C * fC_ph_ph + A * fA_ph_ph) * r**2]
    ])


def dr_spherical_metric(sin_theta, cos_theta, t, r, a, k):
    f_r_r = 2.0 - 3.0 * sin_theta**2
    f_r_th = -3.0 * sin_theta * cos_theta
    fC_th_th = 3.0 * sin_theta**2
    fA_th_th = -1.0
    fC_ph_ph = -3.0 * sin_theta**2
    fA_ph_ph = 3.0 * sin_theta**2 - 1.0
    A = pulse_profile_coefficient_a(t, r, a, k)
    B = pulse_profile_coefficient_b(t, r, a, k)
    C = pulse_profile_coefficient_c(t, r, a, k)
    dr_A = dr_pulse_profile_coefficient_a(t, r, a, k)
    dr_B = dr_pulse_profile_coefficient_b(t, r, a, k)
    dr_C = dr_pulse_profile_coefficient_c(t, r, a, k)
    return np.array([[0.0, 0.0, 0.0, 0.0],
                     [0.0, f_r_r * dr_A, (B + r * dr_B) * f_r_th, 0.0],
                     [
                         0.0, (B + r * dr_B) * f_r_th,
                         r * (2.0 + (2.0 * C + r * dr_C) * fC_th_th +
                              (2.0 * A + r * dr_A) * fA_th_th), 0.0
                     ],
                     [
                         0.0, 0.0, 0.0,
                         r * (2.0 + (2.0 * C + r * dr_C) * fC_ph_ph +
                              (2.0 * A + r * dr_A) * fA_ph_ph)
                     ]])


def dt_spherical_metric(sin_theta, cos_theta, t, r, a, k):
    f_r_r = 2.0 - 3.0 * sin_theta**2
    f_r_th = -3.0 * sin_theta * cos_theta
    fC_th_th = 3.0 * sin_theta**2
    fA_th_th = -1.0
    fC_ph_ph = -3.0 * sin_theta**2
    fA_ph_ph = 3.0 * sin_theta**2 - 1.0
    dt_A = dt_pulse_profile_coefficient_a(t, r, a, k)
    dt_B = dt_pulse_profile_coefficient_b(t, r, a, k)
    dt_C = dt_pulse_profile_coefficient_c(t, r, a, k)
    return np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, f_r_r * dt_A, dt_B * f_r_th * r, 0.0],
         [
             0.0, dt_B * f_r_th * r,
             (dt_C * fC_th_th + dt_A * fA_th_th) * r**2, 0.0
         ], [0.0, 0.0, 0.0, (dt_C * fC_ph_ph + dt_A * fA_ph_ph) * r**2]])


def news(sin_theta, t, r, a, k):
    u = t - r
    return -complex(6.0, 0.0) * a * sin_theta**2 * np.exp(
        -u**2 / k**2) * u / k**10 * (15.0 * k**4 - 20 * k**2 * u**2 +
                                     4.0 * u**4)
