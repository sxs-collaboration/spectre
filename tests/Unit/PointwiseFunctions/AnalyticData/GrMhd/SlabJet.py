# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def piecewise(x, inlet_radius, ambient_value, jet_value):
    return (
        jet_value
        if x[0] <= 0.0 and abs(x[1]) <= inlet_radius
        else ambient_value
    )


def piecewise_vector(x, inlet_radius, ambient_value, jet_value):
    return np.array(
        [
            (
                jet_value[d]
                if x[0] <= 0.0 and abs(x[1]) <= inlet_radius
                else ambient_value[d]
            )
            for d in range(3)
        ]
    )


def parse_vars(*args, **kwargs):
    assert not kwargs, "Found unexpected labeled arguments:\n{}".format(kwargs)
    assert len(args) == 10, "Expected 10 arguments, but got {}".format(
        len(args)
    )
    return dict(
        zip(
            [
                "adiabatic_index",
                "ambient_density",
                "ambient_pressure",
                "ambient_electron_fraction",
                "jet_density",
                "jet_pressure",
                "jet_electron_fraction",
                "jet_velocity",
                "inlet_radius",
                "magnetic_field",
            ],
            args,
        )
    )


def rest_mass_density(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise(
        x, vars["inlet_radius"], vars["ambient_density"], vars["jet_density"]
    )


def electron_fraction(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise(
        x,
        vars["inlet_radius"],
        vars["ambient_electron_fraction"],
        vars["jet_electron_fraction"],
    )


def spatial_velocity(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise_vector(
        x, vars["inlet_radius"], [0.0, 0.0, 0.0], vars["jet_velocity"]
    )


def specific_internal_energy(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    p = pressure(x, *args, **kwargs)
    rho = rest_mass_density(x, *args, **kwargs)
    return p / ((vars["adiabatic_index"] - 1.0) * rho)


def pressure(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise(
        x, vars["inlet_radius"], vars["ambient_pressure"], vars["jet_pressure"]
    )


def specific_enthalpy(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    e = specific_internal_energy(x, *args, **kwargs)
    return 1.0 + vars["adiabatic_index"] * e


def lorentz_factor(x, *args, **kwargs):
    v = spatial_velocity(x, *args, **kwargs)
    return 1.0 / np.sqrt(1.0 - np.sum(v**2))


def magnetic_field(x, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return np.array(vars["magnetic_field"])


def divergence_cleaning_field(x, *args, **kwargs):
    return 0.0
