# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for testing KerrHorizon.cpp

def kerr_horizon_radius(theta, phi, mass, dimless_spin_magnitude,
                        spin_theta, spin_phi):
    spin_a = mass*dimless_spin_magnitude* \
             np.array((np.sin(spin_theta)*np.cos(spin_phi),
                       np.sin(spin_theta)*np.sin(spin_phi),
                       np.cos(spin_theta)))
    a_squared = np.dot(spin_a,spin_a)
    r_plus    = mass+np.sqrt(mass**2-a_squared)
    n_hat     = np.array((np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)))
    return np.sqrt(r_plus**2*(r_plus**2+a_squared)/ \
                   (r_plus**2+np.dot(spin_a,n_hat)**2))

# End functions for testing KerrHorizon.cpp
