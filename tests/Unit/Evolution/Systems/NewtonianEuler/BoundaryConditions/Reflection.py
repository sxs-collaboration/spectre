# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.NewtonianEuler.TestFunctions as TestFunctions
import Evolution.Systems.NewtonianEuler.TimeDerivative as Flux
import numpy as np


def error(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return None


def velocity(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    velocity_dot_normal = np.einsum("i,i", normal_covector, int_velocity)
    velocity = int_velocity - 2.0 * velocity_dot_normal * normal_covector

    if face_mesh_velocity is None:
        return velocity
    else:
        face_mesh_velocity_dot_normal = np.einsum(
            "i,i", normal_covector, face_mesh_velocity
        )
        return velocity + 2.0 * face_mesh_velocity_dot_normal * normal_covector


def mass_density_cons(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return int_mass_density


def momentum_density(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return TestFunctions.momentum_density(
        int_mass_density,
        velocity(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        int_specific_internal_energy,
    )


def energy_density(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return TestFunctions.energy_density(
        int_mass_density,
        velocity(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        int_specific_internal_energy,
    )


def flux_mass_density(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return Flux.mass_density_cons_flux_impl(
        momentum_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        energy_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        velocity(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        int_pressure,
    )


def flux_momentum_density(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return Flux.momentum_density_flux_impl(
        momentum_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        energy_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        velocity(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        int_pressure,
    )


def flux_energy_density(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return Flux.energy_density_flux_impl(
        momentum_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        energy_density(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        velocity(
            face_mesh_velocity,
            normal_covector,
            int_mass_density,
            int_velocity,
            int_specific_internal_energy,
            int_pressure,
        ),
        int_pressure,
    )


def specific_internal_energy(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    int_pressure,
):
    return int_specific_internal_energy
