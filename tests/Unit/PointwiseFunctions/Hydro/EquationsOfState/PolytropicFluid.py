# Distributed under the MIT License.
# See LICENSE.txt for details.


def polytropic_pressure_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return polytropic_constant * rest_mass_density**polytropic_exponent


def polytropic_rel_rest_mass_density_from_enthalpy(
    specific_enthalpy, polytropic_constant, polytropic_exponent
):
    return (
        (polytropic_exponent - 1.0)
        / (polytropic_constant * polytropic_exponent)
        * (specific_enthalpy - 1.0)
    ) ** (1.0 / (polytropic_exponent - 1.0))


def polytropic_newt_rest_mass_density_from_enthalpy(
    specific_enthalpy, polytropic_constant, polytropic_exponent
):
    return (
        (polytropic_exponent - 1.0)
        / (polytropic_constant * polytropic_exponent)
        * (specific_enthalpy)
    ) ** (1.0 / (polytropic_exponent - 1.0))


def polytropic_rel_specific_enthalpy_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return 1.0 + polytropic_constant * polytropic_exponent / (
        polytropic_exponent - 1.0
    ) * rest_mass_density ** (polytropic_exponent - 1.0)


def polytropic_newt_specific_enthalpy_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return (
        polytropic_constant
        * polytropic_exponent
        / (polytropic_exponent - 1.0)
        * rest_mass_density ** (polytropic_exponent - 1.0)
    )


def polytropic_specific_internal_energy_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return (
        polytropic_constant
        * rest_mass_density ** (polytropic_exponent - 1.0)
        / (polytropic_exponent - 1.0)
    )


def polytropic_chi_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return (
        polytropic_constant
        * polytropic_exponent
        * rest_mass_density ** (polytropic_exponent - 1.0)
    )


def polytropic_kappa_times_p_over_rho_squared_from_density(
    rest_mass_density, polytropic_constant, polytropic_exponent
):
    return 0.0
