# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

ORIGIN_EXCLUSION_RADIUS = 0.1
AXIS_COORDINATE_TOLERANCE = 1.0e-14
AXIS_LIMIT_OFFSET = 1.0e-8


def _perturbation_and_dt_perturbation(
    centered_coords,
    t,
    amplitude,
    mode,
    parity,
    direction,
    radius,
    width,
):
    radius_from_center = np.sqrt(np.sum(centered_coords**2))
    if radius_from_center <= ORIGIN_EXCLUSION_RADIUS:
        raise RuntimeError(
            "TeukolskyWave cannot be evaluated at points with radius <= 0.1 "
            "from its center because the implementation uses spherical "
            "coordinates, which are singular at the origin."
        )

    cylindrical_radius = np.sqrt(
        centered_coords[0] ** 2 + centered_coords[1] ** 2
    )
    cos_theta = centered_coords[2] / radius_from_center
    sin_theta = cylindrical_radius / radius_from_center
    if cylindrical_radius > AXIS_COORDINATE_TOLERANCE:
        cos_phi = centered_coords[0] / cylindrical_radius
        sin_phi = centered_coords[1] / cylindrical_radius
    else:
        cos_phi = 1.0
        sin_phi = 0.0
    sin_2phi = 2.0 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    y_profile = (
        radius_from_center - radius + (t if direction == "ingoing" else -t)
    )
    minus_two_over_width_squared = -2.0 / width**2
    profile = amplitude * np.exp(-(y_profile**2) / width**2)
    profile_1 = minus_two_over_width_squared * y_profile * profile
    profile_2 = minus_two_over_width_squared * (profile + y_profile * profile_1)
    profile_3 = minus_two_over_width_squared * (
        2.0 * profile_1 + y_profile * profile_2
    )
    profile_4 = minus_two_over_width_squared * (
        3.0 * profile_2 + y_profile * profile_3
    )
    profile_5 = minus_two_over_width_squared * (
        4.0 * profile_3 + y_profile * profile_4
    )

    spherical_metric = np.zeros((3, 3))
    dt_spherical_metric = np.zeros((3, 3))

    if parity == "even":
        if mode == -2:
            angular_rr = sin_theta**2 * sin_2phi
            angular_rtheta = sin_theta * cos_theta * sin_2phi
            angular_rphi = sin_theta * cos_2phi
            angular_theta_theta_1 = (1.0 + cos_theta**2) * sin_2phi
            angular_theta_theta_2 = -sin_2phi
            angular_thetaphi = -cos_theta * cos_2phi
            angular_phi_phi_1 = -angular_theta_theta_1
            angular_phi_phi_2 = cos_theta**2 * sin_2phi
        elif mode == -1:
            angular_rr = 2.0 * sin_theta * cos_theta * sin_phi
            angular_rtheta = (cos_theta**2 - sin_theta**2) * sin_phi
            angular_rphi = cos_theta * cos_phi
            angular_theta_theta_1 = -2.0 * sin_theta * cos_theta * sin_phi
            angular_theta_theta_2 = 0.0
            angular_thetaphi = sin_theta * cos_phi
            angular_phi_phi_1 = -angular_theta_theta_1
            angular_phi_phi_2 = -2.0 * sin_theta * cos_theta * sin_phi
        elif mode == 0:
            angular_rr = 2.0 - 3.0 * sin_theta**2
            angular_rtheta = -3.0 * sin_theta * cos_theta
            angular_rphi = 0.0
            angular_theta_theta_1 = 3.0 * sin_theta**2
            angular_theta_theta_2 = -1.0
            angular_thetaphi = 0.0
            angular_phi_phi_1 = -angular_theta_theta_1
            angular_phi_phi_2 = 3.0 * sin_theta**2 - 1.0
        elif mode == 1:
            angular_rr = 2.0 * sin_theta * cos_theta * cos_phi
            angular_rtheta = (cos_theta**2 - sin_theta**2) * cos_phi
            angular_rphi = -cos_theta * sin_phi
            angular_theta_theta_1 = -2.0 * sin_theta * cos_theta * cos_phi
            angular_theta_theta_2 = 0.0
            angular_thetaphi = -sin_theta * sin_phi
            angular_phi_phi_1 = -angular_theta_theta_1
            angular_phi_phi_2 = -2.0 * sin_theta * cos_theta * cos_phi
        elif mode == 2:
            angular_rr = sin_theta**2 * cos_2phi
            angular_rtheta = sin_theta * cos_theta * cos_2phi
            angular_rphi = -sin_theta * sin_2phi
            angular_theta_theta_1 = (1.0 + cos_theta**2) * cos_2phi
            angular_theta_theta_2 = -cos_2phi
            angular_thetaphi = cos_theta * sin_2phi
            angular_phi_phi_1 = -angular_theta_theta_1
            angular_phi_phi_2 = cos_theta**2 * cos_2phi
        else:
            raise RuntimeError("Unsupported Teukolsky mode")

        radial_a = (
            3.0
            * (
                profile_2
                + (-3.0 * profile_1 + 3.0 * profile / radius_from_center)
                / radius_from_center
            )
            / radius_from_center**3
        )
        radial_b = (
            -(
                -profile_3
                + (
                    3.0 * profile_2
                    + (-6.0 * profile_1 + 6.0 * profile / radius_from_center)
                    / radius_from_center
                )
                / radius_from_center
            )
            / radius_from_center**2
        )
        radial_c = (
            0.25
            * (
                profile_4
                + (
                    -2.0 * profile_3
                    + (
                        9.0 * profile_2
                        + (
                            -21.0 * profile_1
                            + 21.0 * profile / radius_from_center
                        )
                        / radius_from_center
                    )
                    / radius_from_center
                )
                / radius_from_center
            )
            / radius_from_center
        )

        spherical_metric[0, 0] += radial_a * angular_rr
        spherical_metric[0, 1] += radius_from_center * radial_b * angular_rtheta
        spherical_metric[0, 2] += (
            radius_from_center * radial_b * angular_rphi * sin_theta
        )
        spherical_metric[1, 1] += radius_from_center**2 * (
            radial_c * angular_theta_theta_1 + radial_a * angular_theta_theta_2
        )
        spherical_metric[1, 2] += (
            radius_from_center**2
            * (radial_a - 2.0 * radial_c)
            * angular_thetaphi
            * sin_theta
        )
        spherical_metric[2, 2] += (
            radius_from_center**2
            * (radial_c * angular_phi_phi_1 + radial_a * angular_phi_phi_2)
            * sin_theta**2
        )

        propagation_sign = 1.0 if direction == "ingoing" else -1.0
        dt_radial_a = (
            propagation_sign
            * 3.0
            * (
                profile_3
                + (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center)
                / radius_from_center
            )
            / radius_from_center**3
        )
        dt_radial_b = (
            -propagation_sign
            * (
                -profile_4
                + (
                    3.0 * profile_3
                    + (-6.0 * profile_2 + 6.0 * profile_1 / radius_from_center)
                    / radius_from_center
                )
                / radius_from_center
            )
            / radius_from_center**2
        )
        dt_radial_c = (
            propagation_sign
            * 0.25
            * (
                profile_5
                + (
                    -2.0 * profile_4
                    + (
                        9.0 * profile_3
                        + (
                            -21.0 * profile_2
                            + 21.0 * profile_1 / radius_from_center
                        )
                        / radius_from_center
                    )
                    / radius_from_center
                )
                / radius_from_center
            )
            / radius_from_center
        )

        dt_spherical_metric[0, 0] += dt_radial_a * angular_rr
        dt_spherical_metric[0, 1] += (
            radius_from_center * dt_radial_b * angular_rtheta
        )
        dt_spherical_metric[0, 2] += (
            radius_from_center * dt_radial_b * angular_rphi * sin_theta
        )
        dt_spherical_metric[1, 1] += radius_from_center**2 * (
            dt_radial_c * angular_theta_theta_1
            + dt_radial_a * angular_theta_theta_2
        )
        dt_spherical_metric[1, 2] += (
            radius_from_center**2
            * (dt_radial_a - 2.0 * dt_radial_c)
            * angular_thetaphi
            * sin_theta
        )
        dt_spherical_metric[2, 2] += (
            radius_from_center**2
            * (
                dt_radial_c * angular_phi_phi_1
                + dt_radial_a * angular_phi_phi_2
            )
            * sin_theta**2
        )
    else:
        if mode == -2:
            angular_rtheta = 4.0 * sin_theta * sin_2phi
            angular_rphi = 4.0 * sin_theta * cos_theta * cos_2phi
            angular_theta_theta = -2.0 * cos_theta * sin_2phi
            angular_thetaphi = -(2.0 - sin_theta**2) * cos_2phi
            angular_phi_phi = 2.0 * cos_theta * sin_2phi
        elif mode == -1:
            angular_rtheta = -2.0 * cos_theta * sin_phi
            angular_rphi = -2.0 * (cos_theta**2 - sin_theta**2) * cos_phi
            angular_theta_theta = -sin_theta * sin_phi
            angular_thetaphi = -cos_theta * sin_theta * cos_phi
            angular_phi_phi = sin_theta * sin_phi
        elif mode == 0:
            angular_rtheta = 0.0
            angular_rphi = -4.0 * cos_theta * sin_theta
            angular_theta_theta = 0.0
            angular_thetaphi = -(sin_theta**2)
            angular_phi_phi = 0.0
        elif mode == 1:
            angular_rtheta = -2.0 * cos_theta * cos_phi
            angular_rphi = 2.0 * (cos_theta**2 - sin_theta**2) * sin_phi
            angular_theta_theta = -sin_theta * cos_phi
            angular_thetaphi = cos_theta * sin_theta * sin_phi
            angular_phi_phi = sin_theta * cos_phi
        elif mode == 2:
            angular_rtheta = 4.0 * sin_theta * cos_2phi
            angular_rphi = -4.0 * sin_theta * cos_theta * sin_2phi
            angular_theta_theta = -2.0 * cos_theta * cos_2phi
            angular_thetaphi = (2.0 - sin_theta**2) * sin_2phi
            angular_phi_phi = 2.0 * cos_theta * cos_2phi
        else:
            raise RuntimeError("Unsupported Teukolsky mode")

        radial_k = (
            profile_2
            + (-3.0 * profile_1 + 3.0 * profile / radius_from_center)
            / radius_from_center
        ) / radius_from_center**2
        radial_l = (
            -profile_3
            + (
                2.0 * profile_2
                + (-3.0 * profile_1 + 3.0 * profile / radius_from_center)
                / radius_from_center
            )
            / radius_from_center
        ) / radius_from_center

        spherical_metric[0, 1] += radius_from_center * radial_k * angular_rtheta
        spherical_metric[0, 2] += (
            radius_from_center * radial_k * angular_rphi * sin_theta
        )
        spherical_metric[1, 1] += (
            radius_from_center**2 * radial_l * angular_theta_theta
        )
        spherical_metric[1, 2] += (
            radius_from_center**2 * radial_l * angular_thetaphi * sin_theta
        )
        spherical_metric[2, 2] += (
            radius_from_center**2
            * radial_l
            * angular_phi_phi
            * sin_theta**2
        )

        propagation_sign = 1.0 if direction == "ingoing" else -1.0
        dt_radial_k = (
            propagation_sign
            * (
                profile_3
                + (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center)
                / radius_from_center
            )
            / radius_from_center**2
        )
        dt_radial_l = (
            propagation_sign
            * (
                -profile_4
                + (
                    2.0 * profile_3
                    + (-3.0 * profile_2 + 3.0 * profile_1 / radius_from_center)
                    / radius_from_center
                )
                / radius_from_center
            )
            / radius_from_center
        )

        dt_spherical_metric[0, 1] += (
            radius_from_center * dt_radial_k * angular_rtheta
        )
        dt_spherical_metric[0, 2] += (
            radius_from_center * dt_radial_k * angular_rphi * sin_theta
        )
        dt_spherical_metric[1, 1] += (
            radius_from_center**2 * dt_radial_l * angular_theta_theta
        )
        dt_spherical_metric[1, 2] += (
            radius_from_center**2 * dt_radial_l * angular_thetaphi * sin_theta
        )
        dt_spherical_metric[2, 2] += (
            radius_from_center**2
            * dt_radial_l
            * angular_phi_phi
            * sin_theta**2
        )

    spherical_metric = spherical_metric + np.triu(spherical_metric, 1).T
    dt_spherical_metric = (
        dt_spherical_metric + np.triu(dt_spherical_metric, 1).T
    )

    sin_theta_for_phi = max(sin_theta, AXIS_COORDINATE_TOLERANCE)
    inverse_jacobian = np.array(
        [
            [
                sin_theta * cos_phi,
                sin_theta * sin_phi,
                cos_theta,
            ],
            [
                cos_theta * cos_phi / radius_from_center,
                cos_theta * sin_phi / radius_from_center,
                -sin_theta / radius_from_center,
            ],
            [
                -sin_phi / (radius_from_center * sin_theta_for_phi),
                cos_phi / (radius_from_center * sin_theta_for_phi),
                0.0,
            ],
        ]
    )
    perturbation = np.einsum(
        "ai,bj,ab->ij", inverse_jacobian, inverse_jacobian, spherical_metric
    )
    dt_perturbation = np.einsum(
        "ai,bj,ab->ij", inverse_jacobian, inverse_jacobian, dt_spherical_metric
    )
    return perturbation, dt_perturbation


def _pointwise_metric(
    x,
    t,
    amplitude,
    mode,
    parity,
    direction,
    center,
    radius,
    width,
    include_minkowski_background,
):
    spatial_metric = np.zeros((3, 3))
    if include_minkowski_background:
        spatial_metric += np.eye(3)
    dt_spatial_metric = np.zeros((3, 3))

    centered_coords = np.asarray(x) - np.asarray(center)
    radius_from_center = np.sqrt(np.sum(centered_coords**2))
    if radius_from_center <= ORIGIN_EXCLUSION_RADIUS:
        raise RuntimeError(
            "TeukolskyWave cannot be evaluated at points with radius <= 0.1 "
            "from its center because the implementation uses spherical "
            "coordinates, which are singular at the origin."
        )

    cylindrical_radius = np.sqrt(
        centered_coords[0] ** 2 + centered_coords[1] ** 2
    )
    if cylindrical_radius > AXIS_COORDINATE_TOLERANCE:
        perturbation, dt_perturbation = _perturbation_and_dt_perturbation(
            centered_coords,
            t,
            amplitude,
            mode,
            parity,
            direction,
            radius,
            width,
        )
    else:
        perturbation_x, dt_perturbation_x = _perturbation_and_dt_perturbation(
            centered_coords + np.array([AXIS_LIMIT_OFFSET, 0.0, 0.0]),
            t,
            amplitude,
            mode,
            parity,
            direction,
            radius,
            width,
        )
        perturbation_y, dt_perturbation_y = _perturbation_and_dt_perturbation(
            centered_coords + np.array([0.0, AXIS_LIMIT_OFFSET, 0.0]),
            t,
            amplitude,
            mode,
            parity,
            direction,
            radius,
            width,
        )
        perturbation = 0.5 * (perturbation_x + perturbation_y)
        dt_perturbation = 0.5 * (dt_perturbation_x + dt_perturbation_y)

    return spatial_metric + perturbation, dt_spatial_metric + dt_perturbation


def _variables(
    x,
    t,
    amplitude,
    mode,
    parity,
    direction,
    center,
    radius,
    width,
    include_minkowski_background,
):
    spatial_metric, dt_spatial_metric = _pointwise_metric(
        np.asarray(x, dtype=float),
        t,
        amplitude,
        mode,
        parity,
        direction,
        center,
        radius,
        width,
        include_minkowski_background,
    )
    lapse = 1.0 if include_minkowski_background else 0.0
    dt_lapse = 0.0
    shift = np.zeros(3)
    dt_shift = np.zeros(3)

    result = {
        "Lapse": lapse,
        "dt(Lapse)": dt_lapse,
        "Shift": shift,
        "dt(Shift)": dt_shift,
        "SpatialMetric": spatial_metric,
        "dt(SpatialMetric)": dt_spatial_metric,
    }

    if include_minkowski_background:
        extrinsic_curvature = -0.5 * dt_spatial_metric
        result["SqrtDetSpatialMetric"] = np.sqrt(np.linalg.det(spatial_metric))
        result["ExtrinsicCurvature"] = extrinsic_curvature
        result["InverseSpatialMetric"] = np.linalg.inv(spatial_metric)

    return result


def teukolsky_wave_variables(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return _variables(
        x,
        t,
        amplitude,
        mode,
        parity,
        direction,
        center,
        radius,
        width,
        True,
    )


def teukolsky_wave_variables_no_background(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return _variables(
        x,
        t,
        amplitude,
        mode,
        parity,
        direction,
        center,
        radius,
        width,
        False,
    )


def teukolsky_wave_lapse(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["Lapse"]


def teukolsky_wave_dt_lapse(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(Lapse)"]


def teukolsky_wave_shift(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["Shift"]


def teukolsky_wave_dt_shift(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(Shift)"]


def teukolsky_wave_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["SpatialMetric"]


def teukolsky_wave_dt_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(SpatialMetric)"]


def teukolsky_wave_sqrt_det_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["SqrtDetSpatialMetric"]


def teukolsky_wave_extrinsic_curvature(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["ExtrinsicCurvature"]


def teukolsky_wave_inverse_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["InverseSpatialMetric"]


def teukolsky_wave_no_background_lapse(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["Lapse"]


def teukolsky_wave_no_background_dt_lapse(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(Lapse)"]


def teukolsky_wave_no_background_shift(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["Shift"]


def teukolsky_wave_no_background_dt_shift(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(Shift)"]


def teukolsky_wave_no_background_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["SpatialMetric"]


def teukolsky_wave_no_background_dt_spatial_metric(
    x, t, amplitude, mode, parity, direction, center, radius, width
):
    return teukolsky_wave_variables_no_background(
        x, t, amplitude, mode, parity, direction, center, radius, width
    )["dt(SpatialMetric)"]
