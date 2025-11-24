# Distributed under the MIT License.
# See LICENSE.txt for details.


def relativistic_specific_enthalpy(
    rest_mass_density, specific_internal_energy, pressure
):
    return pressure / rest_mass_density + 1.0 + specific_internal_energy
