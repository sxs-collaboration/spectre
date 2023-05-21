# Distributed under the MIT License.
# See LICENSE.txt for details.


def calc_eint_constant(
    polytropic_exponent,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    eint_constant = (
        (polytropic_exponent - poly_exponent_lo)
        / ((poly_exponent_hi - 1.0) * (poly_exponent_lo - 1.0))
        * poly_constant_lo
        * transition_density ** (poly_exponent_lo - 1.0)
    )
    return eint_constant


def piecewisepolytropic_pressure_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    if rest_mass_density < transition_density:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi
    return polytropic_constant * rest_mass_density**polytropic_exponent


def piecewisepolytropic_rel_rest_mass_density_from_enthalpy(
    specific_enthalpy,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    transition_pressure = poly_constant_lo * transition_density ** (
        poly_exponent_lo
    )
    transition_spec_eint = transition_pressure / (
        (poly_exponent_lo - 1.0) * transition_density
    )
    transition_spec_enthalpy = (
        1.0 + transition_spec_eint + transition_pressure / transition_density
    )

    if specific_enthalpy < transition_spec_enthalpy:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi

    eint_constant = calc_eint_constant(
        polytropic_exponent,
        transition_density,
        poly_constant_lo,
        poly_exponent_lo,
        poly_exponent_hi,
    )

    return (
        (polytropic_exponent - 1.0)
        / (polytropic_constant * polytropic_exponent)
        * (specific_enthalpy - 1.0 - eint_constant)
    ) ** (1.0 / (polytropic_exponent - 1.0))


def piecewisepolytropic_newt_rest_mass_density_from_enthalpy(
    specific_enthalpy,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    transition_pressure = poly_constant_lo * transition_density ** (
        poly_exponent_lo
    )
    transition_spec_eint = transition_pressure / (
        (poly_exponent_lo - 1.0) * transition_density
    )
    transition_spec_enthalpy = (
        transition_spec_eint + transition_pressure / transition_density
    )

    if specific_enthalpy < transition_spec_enthalpy:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi

    eint_constant = calc_eint_constant(
        polytropic_exponent,
        transition_density,
        poly_constant_lo,
        poly_exponent_lo,
        poly_exponent_hi,
    )

    return (
        (polytropic_exponent - 1.0)
        / (polytropic_constant * polytropic_exponent)
        * (specific_enthalpy - eint_constant)
    ) ** (1.0 / (polytropic_exponent - 1.0))


def piecewisepolytropic_rel_specific_enthalpy_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    if rest_mass_density < transition_density:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi

    eint_constant = calc_eint_constant(
        polytropic_exponent,
        transition_density,
        poly_constant_lo,
        poly_exponent_lo,
        poly_exponent_hi,
    )

    return (
        1.0
        + polytropic_constant
        * polytropic_exponent
        / (polytropic_exponent - 1.0)
        * rest_mass_density ** (polytropic_exponent - 1.0)
        + eint_constant
    )


def piecewisepolytropic_newt_specific_enthalpy_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    if rest_mass_density < transition_density:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi

    eint_constant = calc_eint_constant(
        polytropic_exponent,
        transition_density,
        poly_constant_lo,
        poly_exponent_lo,
        poly_exponent_hi,
    )

    return (
        polytropic_constant
        * polytropic_exponent
        / (polytropic_exponent - 1.0)
        * rest_mass_density ** (polytropic_exponent - 1.0)
        + eint_constant
    )


def piecewisepolytropic_specific_internal_energy_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    if rest_mass_density < transition_density:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi

    eint_constant = calc_eint_constant(
        polytropic_exponent,
        transition_density,
        poly_constant_lo,
        poly_exponent_lo,
        poly_exponent_hi,
    )

    return (
        polytropic_constant
        * rest_mass_density ** (polytropic_exponent - 1.0)
        / (polytropic_exponent - 1.0)
        + eint_constant
    )


def piecewisepolytropic_chi_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    poly_constant_hi = poly_constant_lo * transition_density ** (
        poly_exponent_lo - poly_exponent_hi
    )
    if rest_mass_density < transition_density:
        polytropic_constant = poly_constant_lo
        polytropic_exponent = poly_exponent_lo
    else:
        polytropic_constant = poly_constant_hi
        polytropic_exponent = poly_exponent_hi
    return (
        polytropic_constant
        * polytropic_exponent
        * rest_mass_density ** (polytropic_exponent - 1.0)
    )


def piecewisepolytropic_kappa_times_p_over_rho_squared_from_density(
    rest_mass_density,
    transition_density,
    poly_constant_lo,
    poly_exponent_lo,
    poly_exponent_hi,
):
    return 0.0
