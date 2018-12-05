# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for Test_KomissarovShock.cpp

def parse_vars(*args, **kwargs):
    assert not kwargs, "Found unexpected labeled arguments:\n{}".format(kwargs)
    assert len(args) is 10, "Expected 10 arguments, but got {}".format(
        len(args))
    return dict(zip(['adiabatic_index', 'left_rest_mass_density',
            'right_rest_mass_density', 'left_pressure', 'right_pressure',
            'left_spatial_velocity', 'right_spatial_velocity',
            'left_magnetic_field', 'right_magnetic_field', 'shock_speed'],
            args))

def piecewise(x, shock_position, left_value, right_value):
    return left_value if x[0] < shock_position else right_value

def rest_mass_density(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise(x, t * vars['shock_speed'],
        vars['left_rest_mass_density'], vars['right_rest_mass_density'])

def spatial_velocity(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return np.asarray(piecewise(x, t * vars['shock_speed'],
        vars['left_spatial_velocity'], vars['right_spatial_velocity']))

def pressure(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return piecewise(x, t * vars['shock_speed'],
        vars['left_pressure'], vars['right_pressure'])

def specific_internal_energy(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    p = pressure(x, t, *args, **kwargs)
    rho = rest_mass_density(x, t, *args, **kwargs)
    return p / ((vars['adiabatic_index'] - 1.0) * rho)

def specific_enthalpy(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    e = specific_internal_energy(x, t, *args, **kwargs)
    return 1.0 + vars['adiabatic_index'] * e

def lorentz_factor(x, t, *args, **kwargs):
    v = spatial_velocity(x, t, *args, **kwargs)
    return 1.0 / np.sqrt(1.0 - np.sum(v**2))

def magnetic_field(x, t, *args, **kwargs):
    vars = parse_vars(*args, **kwargs)
    return np.asarray(piecewise(x, t * vars['shock_speed'],
        vars['left_magnetic_field'], vars['right_magnetic_field']))

def divergence_cleaning_field(x, t, *args, **kwargs):
    return 0.0
