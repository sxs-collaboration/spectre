# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def self_force_acceleration(
    dt_psi_monopole, psi_dipole, vel, charge, mass, inverse_metric, dilation
):
    # Prepend extra value so dimensions work out for einsum.
    # The 0th component does not affect the final result
    four_vel = np.concatenate((np.empty(1), vel), axis=0)
    d_psi = np.concatenate(([dt_psi_monopole], psi_dipole), axis=0)
    self_force_acc = np.einsum("ij,j", inverse_metric, d_psi)
    self_force_acc -= np.einsum("i,j,j", four_vel, inverse_metric[0], d_psi)
    self_force_acc *= charge / mass / dilation**2
    return self_force_acc[1:]


def self_force_per_mass(d_psi, four_velocity, charge, mass, inverse_metric):
    self_force_per_mass = np.einsum("ij,j", inverse_metric, d_psi)
    self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, d_psi
    )
    return charge / mass * self_force_per_mass


def dt_self_force_per_mass(
    d_psi,
    dt_d_psi,
    four_velocity,
    dt_four_velocity,
    charge,
    mass,
    inverse_metric,
    dt_inverse_metric,
):
    dt_self_force_per_mass = np.einsum("ij,j", dt_inverse_metric, d_psi)
    dt_self_force_per_mass += np.einsum("ij,j", inverse_metric, dt_d_psi)

    dt_self_force_per_mass += np.einsum(
        "i,j,j", dt_four_velocity, four_velocity, d_psi
    )
    dt_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, dt_four_velocity, d_psi
    )
    dt_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, dt_d_psi
    )
    return charge / mass * dt_self_force_per_mass


def dt2_self_force_per_mass(
    d_psi,
    dt_d_psi,
    dt2_d_psi,
    four_velocity,
    dt_four_velocity,
    dt2_four_velocity,
    charge,
    mass,
    inverse_metric,
    dt_inverse_metric,
    dt2_inverse_metric,
):
    dt2_self_force_per_mass = np.einsum("ij,j", dt2_inverse_metric, d_psi)
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "ij,j", dt_inverse_metric, dt_d_psi
    )
    dt2_self_force_per_mass += np.einsum("ij,j", inverse_metric, dt2_d_psi)

    dt2_self_force_per_mass += np.einsum(
        "i,j,j", dt2_four_velocity, four_velocity, d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", dt_four_velocity, dt_four_velocity, d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", dt_four_velocity, four_velocity, dt_d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", four_velocity, dt_four_velocity, dt_d_psi
    )
    dt2_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, dt2_four_velocity, d_psi
    )
    dt2_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, dt2_d_psi
    )
    return charge / mass * dt2_self_force_per_mass


def Du_self_force_per_mass(
    self_force, dt_self_force, four_velocity, christoffel
):
    Du_self_force_per_mass = four_velocity[0] * dt_self_force
    Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, four_velocity, self_force
    )
    return Du_self_force_per_mass


def dt_Du_self_force_per_mass(
    self_force,
    dt_self_force,
    dt2_self_force,
    four_velocity,
    dt_four_velocity,
    christoffel,
    dt_christoffel,
):
    dt_Du_self_force_per_mass = (
        dt_four_velocity[0] * dt_self_force + dt2_self_force * four_velocity[0]
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", dt_christoffel, four_velocity, self_force
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, dt_four_velocity, self_force
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, four_velocity, dt_self_force
    )
    return dt_Du_self_force_per_mass
