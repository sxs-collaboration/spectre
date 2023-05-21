# Distributed under the MIT License.
# See LICENSE.txt for details.

import math

import numpy as np


def cartesian_to_angular_coordinates(
    cos_phi, cos_theta, sin_phi, sin_theta, extraction_radius
):
    return np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])


def cartesian_to_angular_jacobian(
    cos_phi, cos_theta, sin_phi, sin_theta, extraction_radius
):
    return np.array(
        [
            [cos_phi * sin_theta, sin_phi * sin_theta, cos_theta],
            [
                extraction_radius * cos_phi * cos_theta,
                extraction_radius * sin_phi * cos_theta,
                -extraction_radius * sin_theta,
            ],
            [-extraction_radius * sin_phi, extraction_radius * cos_phi, 0.0],
        ]
    )


def cartesian_to_angular_inverse_jacobian(
    cos_phi, cos_theta, sin_phi, sin_theta, extraction_radius
):
    return np.array(
        [
            [
                cos_phi * sin_theta,
                cos_phi * cos_theta / extraction_radius,
                -sin_phi / extraction_radius,
            ],
            [
                sin_phi * sin_theta,
                cos_theta * sin_phi / extraction_radius,
                cos_phi / extraction_radius,
            ],
            [cos_theta, -sin_theta / extraction_radius, 0.0],
        ]
    )


def null_metric(cartesian_to_angular_jacobian, pi, psi):
    null_metric = psi.copy()
    null_metric[1, 1] = 0.0

    null_metric[1, 2:4] = 0.0
    null_metric[2:4, 1] = 0.0

    null_metric[0, 1] = -1.0
    null_metric[1, 0] = -1.0

    null_metric[0, 0] = psi[0, 0]

    null_metric[0, 2:4] = np.einsum(
        "Ai,i", cartesian_to_angular_jacobian, psi[0, 1:4]
    )[1:3]
    null_metric[2:4, 0] = null_metric[0, 2:4]

    null_metric[2:4, 2:4] = np.einsum(
        "Ai,Bj,ij",
        cartesian_to_angular_jacobian,
        cartesian_to_angular_jacobian,
        psi[1:4, 1:4],
    )[1:4, 1:4]
    return null_metric


def du_null_metric(cartesian_to_angular_jacobian, pi, psi):
    du_null_metric = pi.copy()
    du_null_metric[1, 0:4] = 0.0
    du_null_metric[0:4, 1] = 0.0

    du_null_metric[0, 0] = pi[0, 0]

    du_null_metric[0, 2:4] = np.einsum(
        "Ai,i", cartesian_to_angular_jacobian, pi[0][1:4]
    )[1:3]
    du_null_metric[2:4, 0] = du_null_metric[0, 2:4]

    du_null_metric[2:4, 2:4] = np.einsum(
        "Ai,Bj,ij",
        cartesian_to_angular_jacobian,
        cartesian_to_angular_jacobian,
        pi[1:4, 1:4],
    )[1:4, 1:4]

    return du_null_metric


def inverse_null_metric(null_metric):
    inverse_null_metric = null_metric.copy()
    inverse_null_metric[1, 0] = -1.0
    inverse_null_metric[0, 1] = -1.0

    inverse_null_metric[0, 0] = 0.0
    inverse_null_metric[0, 2:4] = 0.0
    inverse_null_metric[2:4, 0] = 0.0
    angular_determinant = (
        null_metric[2, 2] * null_metric[3, 3]
        - null_metric[2, 3] * null_metric[3, 2]
    )
    inverse_null_metric[2, 2] = null_metric[3, 3] / angular_determinant
    inverse_null_metric[2, 3] = -null_metric[2, 3] / angular_determinant
    inverse_null_metric[3, 2] = -null_metric[2, 3] / angular_determinant
    inverse_null_metric[3, 3] = null_metric[2, 2] / angular_determinant

    inverse_null_metric[1, 2:4] = np.einsum(
        "AB,B", inverse_null_metric[2:4, 2:4], null_metric[2:4, 0]
    )
    inverse_null_metric[2:4, 1] = inverse_null_metric[1, 2:4]
    inverse_null_metric[1, 1] = -null_metric[0, 0] + np.einsum(
        "A,A", inverse_null_metric[1, 2:4], null_metric[2:4, 0]
    )
    return inverse_null_metric


def worldtube_normal(
    cos_phi, cos_theta, psi, dt_psi, sin_phi, sin_theta, inverse_spatial_metric
):
    sigma = inverse_spatial_metric[0, :].copy()
    sigma[0] = cos_phi * sin_theta**2
    sigma[1] = sin_phi * sin_theta**2
    sigma[2] = cos_theta * sin_theta
    norm_of_sigma = math.sqrt(
        np.einsum("i,j,ij", sigma, sigma, inverse_spatial_metric)
    )

    worldtube_normal = np.einsum(
        "ij,j", inverse_spatial_metric, sigma / norm_of_sigma
    )
    return worldtube_normal


def dt_worldtube_normal(
    cos_phi, cos_theta, psi, dt_psi, sin_phi, sin_theta, inverse_spatial_metric
):
    sigma = inverse_spatial_metric[0, :].copy()
    sigma[0] = cos_phi * sin_theta**2
    sigma[1] = sin_phi * sin_theta**2
    sigma[2] = cos_theta * sin_theta
    norm_of_sigma = math.sqrt(
        np.einsum("i,j,ij", sigma, sigma, inverse_spatial_metric)
    )

    worldtube_normal = np.einsum(
        "ij,j", inverse_spatial_metric, sigma / norm_of_sigma
    )

    dt_worldtube_normal = np.einsum(
        "ij,k,jk",
        0.5 * np.outer(worldtube_normal, worldtube_normal)
        - inverse_spatial_metric,
        worldtube_normal,
        dt_psi[1:4, 1:4],
    )
    return dt_worldtube_normal


def null_vector_l(
    dt_worldtube_normal,
    dt_lapse,
    dt_psi,
    dt_shift,
    lapse,
    psi,
    shift,
    worldtube_normal,
):
    hypersurface_normal_vector = np.pad(
        worldtube_normal, ((1, 0)), "constant"
    ).copy()
    hypersurface_normal_vector[0] = 1.0 / lapse
    hypersurface_normal_vector[1:4] = -shift / lapse
    null_l = np.pad(worldtube_normal, ((1, 0)), "constant").copy()
    null_l[0] = hypersurface_normal_vector[0] / (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal)
    )
    null_l[1:4] = (hypersurface_normal_vector[1:4] + worldtube_normal) / (
        lapse - np.einsum("ij,i,j", psi[1:4, 1:4], shift, worldtube_normal)
    )
    return null_l


def du_null_vector_l(
    dt_worldtube_normal,
    dt_lapse,
    dt_psi,
    dt_shift,
    lapse,
    psi,
    shift,
    worldtube_normal,
):
    hypersurface_normal_vector = np.pad(
        worldtube_normal, ((1, 0)), "constant"
    ).copy()
    hypersurface_normal_vector[0] = 1.0 / lapse
    hypersurface_normal_vector[1:4] = -shift / lapse

    denominator = lapse - np.einsum(
        "ij,i,j", psi[1:4, 1:4], shift, worldtube_normal
    )

    du_hypersurface_normal = np.pad(shift[:], ((1, 0)), "constant").copy()
    du_hypersurface_normal[0] = -dt_lapse / lapse**2
    du_hypersurface_normal[1:4] = -(dt_shift / lapse) + (
        np.outer(dt_lapse, shift) / lapse**2
    )

    du_null_l = (du_hypersurface_normal) / denominator
    du_null_l[1:4] += (dt_worldtube_normal) / denominator
    du_denominator = (
        -dt_lapse
        + np.einsum("ij,i,j", dt_psi[1:4, 1:4], shift, worldtube_normal)
        + np.einsum("ij,i,j", psi[1:4, 1:4], dt_shift, worldtube_normal)
        + np.einsum("ij,i,j", psi[1:4, 1:4], shift, dt_worldtube_normal)
    )

    du_null_l[0] += (
        du_denominator * hypersurface_normal_vector[0] / denominator**2
    )
    du_null_l[1:4] += (
        du_denominator
        * (hypersurface_normal_vector[1:4] + worldtube_normal)
        / denominator**2
    )
    return du_null_l


def dlambda_null_metric(
    angular_d_null_l,
    cartesian_to_angular_jacobian,
    phi,
    pi,
    du_null_l,
    inverse_null_metric,
    null_l,
    psi,
):
    dlambda_null_metric = psi.copy()
    dlambda_null_metric[0, 0] = (
        np.einsum("i,i", null_l[1:4], phi[:, 0, 0])
        + null_l[0] * pi[0, 0]
        + 2.0 * np.einsum("a,a", du_null_l, psi[:, 0])
    )
    dlambda_null_metric[1, :] = 0.0
    dlambda_null_metric[:, 1] = 0.0

    dlambda_null_metric[0, 2:4] = np.einsum(
        "Ak,a,ka", cartesian_to_angular_jacobian[1:3, :], du_null_l, psi[1:4, :]
    )
    dlambda_null_metric[0, 2:4] += null_l[0] * np.einsum(
        "Ak,k", cartesian_to_angular_jacobian[1:3, :], pi[1:4, 0]
    )
    dlambda_null_metric[0, 2:4] += np.einsum(
        "Ak,i,ik",
        cartesian_to_angular_jacobian[1:3, :],
        null_l[1:4],
        phi[:, 1:4, 0],
    )
    dlambda_null_metric[0, 2:4] += np.einsum(
        "Aa,a", angular_d_null_l[0:2, :], psi[:, 0]
    )
    dlambda_null_metric[2:4, 0] = dlambda_null_metric[0, 2:4]

    dlambda_null_metric[2:4, 2:4] = null_l[0] * np.einsum(
        "Ak,Bl,kl",
        cartesian_to_angular_jacobian[1:3, :],
        cartesian_to_angular_jacobian[1:3, :],
        pi[1:4, 1:4],
    )
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Ak,Bl,i,ikl",
        cartesian_to_angular_jacobian[1:3, :],
        cartesian_to_angular_jacobian[1:3, :],
        null_l[1:4],
        phi[:, 1:4, 1:4],
    )
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Aa,Bl,al",
        angular_d_null_l[0:2, :],
        cartesian_to_angular_jacobian[1:3, :],
        psi[:, 1:4],
    )
    dlambda_null_metric[2:4, 2:4] += np.einsum(
        "Al,Ba,al",
        cartesian_to_angular_jacobian[1:3, :],
        angular_d_null_l[0:2, :],
        psi[:, 1:4],
    )
    return dlambda_null_metric


def inverse_dlambda_null_metric(
    angular_d_null_l,
    cartesian_to_angular_jacobian,
    phi,
    pi,
    du_null_l,
    inverse_null_metric,
    null_l,
    psi,
):
    dlambda_null_metric_value = dlambda_null_metric(
        angular_d_null_l,
        cartesian_to_angular_jacobian,
        phi,
        pi,
        du_null_l,
        inverse_null_metric,
        null_l,
        psi,
    )

    inverse_dlambda_null_metric_value = dlambda_null_metric_value.copy()

    inverse_dlambda_null_metric_value[0, :] = 0.0
    inverse_dlambda_null_metric_value[:, 0] = 0.0
    inverse_dlambda_null_metric_value[1, 1] = (
        -dlambda_null_metric_value[0, 0]
        + 2.0
        * np.einsum(
            "a,a",
            inverse_null_metric[1, 2:4],
            dlambda_null_metric_value[0, 2:4],
        )
        - np.einsum(
            "a,b,ab",
            inverse_null_metric[1, 2:4],
            inverse_null_metric[1, 2:4],
            dlambda_null_metric_value[2:4, 2:4],
        )
    )
    inverse_dlambda_null_metric_value[1, 2:4] = np.einsum(
        "ab, b",
        inverse_null_metric[2:4, 2:4],
        dlambda_null_metric_value[0, 2:4],
    ) - np.einsum(
        "ab, c, cb",
        inverse_null_metric[2:4, 2:4],
        inverse_null_metric[1, 2:4],
        dlambda_null_metric_value[2:4, 2:4],
    )
    inverse_dlambda_null_metric_value[2:4, 2:4] = -np.einsum(
        "ac,bd,cd",
        inverse_null_metric[2:4, 2:4],
        inverse_null_metric[2:4, 2:4],
        dlambda_null_metric_value[2:4, 2:4],
    )

    return inverse_dlambda_null_metric_value


def bondi_beta_worldtube_data(local_d_bondi_r):
    return np.array(-0.5 * math.log(local_d_bondi_r[1]))


def bondi_u_worldtube_data(down_dyad, local_d_bondi_r, inverse_null_metric):
    u_vector = local_d_bondi_r[1] * inverse_null_metric[1, 2:4] + np.einsum(
        "b, ab", local_d_bondi_r[2:4], inverse_null_metric[2:4, 2:4]
    )
    return -np.einsum("a,a", u_vector, down_dyad) / local_d_bondi_r[1]


def bondi_w_worldtube_data(local_d_bondi_r, inverse_null_metric, local_bondi_r):
    return (
        -1.0
        + (
            inverse_null_metric[1, 1] * local_d_bondi_r[1] ** 2
            + 2.0
            * local_d_bondi_r[1]
            * (
                np.einsum(
                    "a,a", local_d_bondi_r[2:4], inverse_null_metric[1, 2:4]
                )
                - local_d_bondi_r[0]
            )
            + np.einsum(
                "a,b,ab",
                local_d_bondi_r[2:4],
                local_d_bondi_r[2:4],
                inverse_null_metric[2:4, 2:4],
            )
        )
        / local_d_bondi_r[1]
    ) / local_bondi_r


def bondi_j_worldtube_data(null_metric, bondi_r, up_dyad):
    return (
        0.5
        * np.einsum("a,b,ab", up_dyad, up_dyad, null_metric[2:4, 2:4])
        / bondi_r**2
    )


def dr_bondi_j_worldtube_data(
    dlambda_null_metric, local_d_bondi_r, bondi_j, local_bondi_r, up_dyad
):
    return (
        0.5
        * np.einsum("a,b,ab", up_dyad, up_dyad, dlambda_null_metric[2:4, 2:4])
        / local_bondi_r**2
        - 2.0 * local_d_bondi_r[1] * bondi_j / local_bondi_r
    ) / local_d_bondi_r[1]


def dr_bondi_j_denominator(
    dlambda_null_metric, local_d_bondi_r, bondi_j, local_bondi_r, up_dyad
):
    return 1.0 / (local_bondi_r**2 * local_d_bondi_r[1])


def d2lambda_bondi_r(local_d_bondi_r, dr_bondi_j, bondi_j, bondi_r):
    return np.array(
        np.real(
            -0.25
            * bondi_r
            * local_d_bondi_r[1] ** 2
            * (
                dr_bondi_j * np.conj(dr_bondi_j)
                - 0.25
                * (
                    (
                        dr_bondi_j * np.conj(bondi_j)
                        + bondi_j * np.conj(dr_bondi_j)
                    )
                    / np.sqrt(1.0 + bondi_j * np.conj(bondi_j))
                )
                ** 2
            )
        )
    )


def dlambda_bondi_u(
    d2lambda_bondi_r,
    dlambda_inverse_null_metric,
    local_d_bondi_r,
    down_dyad,
    angular_d_dlambda_r,
    inverse_null_metric,
    bondi_u,
):
    dlambda_u_vector_part = -(
        dlambda_inverse_null_metric[1, 2:4]
        + np.einsum(
            "b,ab",
            angular_d_dlambda_r[0:2] / local_d_bondi_r[1],
            inverse_null_metric[2:4, 2:4],
        )
        + np.einsum(
            "b, ab",
            local_d_bondi_r[2:4] / local_d_bondi_r[1],
            dlambda_inverse_null_metric[2:4, 2:4],
        )
    )
    dlambda_beta = -0.5 * d2lambda_bondi_r / local_d_bondi_r[1]

    return np.einsum(
        "a,a", dlambda_u_vector_part, down_dyad
    ) + 2.0 * dlambda_beta * (
        bondi_u + np.einsum("a,a", inverse_null_metric[1, 2:4], down_dyad)
    )


def bondi_q_worldtube_data(
    local_d2lambda_bondi_r,
    dlambda_inverse_null_metric,
    local_d_bondi_r,
    down_dyad,
    angular_d_dlambda_r,
    inverse_null_metric,
    bondi_j,
    local_bondi_r,
    bondi_u,
):
    local_dlambda_bondi_u = dlambda_bondi_u(
        local_d2lambda_bondi_r,
        dlambda_inverse_null_metric,
        local_d_bondi_r,
        down_dyad,
        angular_d_dlambda_r,
        inverse_null_metric,
        bondi_u,
    )
    return local_bondi_r**2 * (
        bondi_j * np.conj(local_dlambda_bondi_u)
        + np.sqrt(1.0 + bondi_j * np.conj(bondi_j)) * local_dlambda_bondi_u
    )


def dr_bondi_u_worldtube_data(
    local_d2lambda_bondi_r,
    dlambda_inverse_null_metric,
    local_d_bondi_r,
    down_dyad,
    angular_d_dlambda_r,
    inverse_null_metric,
    bondi_j,
    local_bondi_r,
    bondi_u,
):
    local_dlambda_bondi_u = dlambda_bondi_u(
        local_d2lambda_bondi_r,
        dlambda_inverse_null_metric,
        local_d_bondi_r,
        down_dyad,
        angular_d_dlambda_r,
        inverse_null_metric,
        bondi_u,
    )
    return local_dlambda_bondi_u / local_d_bondi_r[1]


def bondi_h_worldtube_data(
    local_d_bondi_r, bondi_j, du_null_metric, local_bondi_r, up_dyad
):
    return (
        0.5
        * np.einsum("a,b,ab", up_dyad, up_dyad, du_null_metric[2:4, 2:4])
        / local_bondi_r**2
        - 2.0 * local_d_bondi_r[0] * bondi_j / local_bondi_r
    )


def du_j_worldtube_data(
    local_d_bondi_r,
    bondi_j,
    du_null_metric,
    dlambda_null_metric,
    local_bondi_r,
    up_dyad,
):
    return (
        0.5
        * np.einsum(
            "a,b,ab",
            up_dyad,
            up_dyad,
            du_null_metric[2:4, 2:4]
            - local_d_bondi_r[0]
            * dlambda_null_metric[2:4, 2:4]
            / local_d_bondi_r[1],
        )
        / local_bondi_r**2
    )
