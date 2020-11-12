# Distributed under the MIT License.
# See LICENSE.txt for details.

from numpy import array, cos, exp, sin


def coordinate_wave_function(t, r, f, a, tpeak, duration):
    u = t - r
    return a * sin(f * u) * exp(-(u - tpeak)**2 / duration**2)


def du_coordinate_wave_function(t, r, f, a, tpeak, duration):
    u = t - r
    return a * (f * cos(f * u) - 2.0 * sin(f * u) *
                (u - tpeak) / duration**2) * exp(-(u - tpeak)**2 / duration**2)


def du_du_coordinate_wave_function(t, r, f, a, tpeak, duration):
    u = t - r
    return (-a * f**2 * sin(f * u) - 2.0 * a * f * cos(f * u) *
            (u - tpeak) / duration**2 - 2.0 * a * sin(f * u) / duration**2 -
            2.0 * (u - tpeak) / duration**2 *
            (a * f * cos(f * u) - 2.0 * a * sin(f * u) *
             (u - tpeak) / duration**2)) * exp(-(u - tpeak)**2 / duration**2)


def spherical_metric(sin_theta, cos_theta, t, r, m, f, a, tpeak, duration):
    wave_func = coordinate_wave_function(t, r, f, a, tpeak, duration)
    du_wave_func = du_coordinate_wave_function(t, r, f, a, tpeak, duration)
    g_tt = -(r - 2.0 * m) * (r + du_wave_func)**2 / r**3
    g_tr = (1 + du_wave_func / r) * (2.0 * m / r + (1.0 - 2.0 * m / r) *
                                     (du_wave_func / r + wave_func / r**2))
    g_rr = (1 - du_wave_func / r -
            wave_func / r**2) * (1.0 + 2.0 * m / r + (1.0 - 2.0 * m / r) *
                                 (du_wave_func / r + wave_func / r**2))
    return array([[g_tt, g_tr, 0.0, 0.0], [g_tr, g_rr, 0.0, 0.0],
                  [0.0, 0.0, r**2, 0.0], [0.0, 0.0, 0.0, r**2]])


def dt_spherical_metric(sin_theta, cos_theta, t, r, m, f, a, tpeak, duration):
    wave_func = coordinate_wave_function(t, r, f, a, tpeak, duration)
    du_wave_func = du_coordinate_wave_function(t, r, f, a, tpeak, duration)
    du_du_wave_func = du_du_coordinate_wave_function(t, r, f, a, tpeak,
                                                     duration)
    dt_g_tt = (-2.0 * (1.0 - 2.0 * m / r) * (1.0 + du_wave_func / r) *
               (du_du_wave_func / r))
    dt_g_tr = ((du_du_wave_func / r) *
               (2.0 * m / r + (1.0 - 2.0 * m / r) *
                (du_wave_func / r + wave_func / r**2)) +
               (1.0 + du_wave_func / r) * (1.0 - 2.0 * m / r) *
               (du_du_wave_func / r + du_wave_func / r**2))
    dt_g_rr = ((-du_du_wave_func / r - du_wave_func / r**2) *
               (1.0 + 2.0 * m / r + (1.0 - 2.0 * m / r) *
                (du_wave_func / r + wave_func / r**2)) +
               (1.0 - du_wave_func / r - wave_func / r**2) *
               (1.0 - 2.0 * m / r) *
               (du_du_wave_func / r + du_wave_func / r**2))
    return array([[dt_g_tt, dt_g_tr, 0.0, 0.0], [dt_g_tr, dt_g_rr, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


def dr_spherical_metric(sin_theta, cos_theta, t, r, m, f, a, tpeak, duration):
    wave_func = coordinate_wave_function(t, r, f, a, tpeak, duration)
    du_wave_func = du_coordinate_wave_function(t, r, f, a, tpeak, duration)
    dt_metric = dt_spherical_metric(sin_theta, cos_theta, t, r, m, f, a, tpeak,
                                    duration)
    dr_g_tt = -(2.0 * m / r**2) * (1.0 + du_wave_func / r)**2 + 2.0 * (
        1.0 - 2.0 * m / r) * (1.0 + du_wave_func / r) * du_wave_func / r**2
    dr_g_tr = (-du_wave_func / r**2 * (2.0 * m / r + (1.0 - 2.0 * m / r) *
                                       (du_wave_func / r + wave_func / r**2)) +
               (1.0 + du_wave_func / r) *
               (-2.0 * m / r**2 + 2.0 * m / r**2 *
                (du_wave_func / r + wave_func / r**2) + (1.0 - 2.0 * m / r) *
                (-du_wave_func / r**2 - 2.0 * wave_func / r**3)))
    dr_g_rr = ((du_wave_func / r**2 + 2.0 * wave_func / r**3) *
               (1.0 + 2.0 * m / r + (1.0 - 2.0 * m / r) *
                (du_wave_func / r + wave_func / r**2)) +
               (1.0 - du_wave_func / r - wave_func / r**2) *
               (-2.0 * m / r**2 + 2.0 * m / r**2 *
                (du_wave_func / r + wave_func / r**2) + (1.0 - 2.0 * m / r) *
                (-du_wave_func / r**2 - 2.0 * wave_func / r**3)))
    return array(
        [[dr_g_tt - dt_metric[0, 0], dr_g_tr - dt_metric[0, 1], 0.0, 0.0],
         [dr_g_tr - dt_metric[1, 0], dr_g_rr - dt_metric[1, 1], 0.0, 0.0],
         [0.0, 0.0, 2.0 * r, 0.0], [0.0, 0.0, 0.0, 2.0 * r]])


def news(sin_theta, t, r, m, f, a, tpeak, duration):
    return complex(0.0, 0.0)
