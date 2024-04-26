# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def initial_orbital_parameters(
    mass_ratio: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    eccentricity: Optional[float] = None,
    mean_anomaly_fraction: Optional[float] = None,
    separation: Optional[float] = None,
    orbital_angular_velocity: Optional[float] = None,
    radial_expansion_velocity: Optional[float] = None,
    num_orbits: Optional[float] = None,
    time_to_merger: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Estimate initial orbital parameters from a Post-Newtonian approximation.

    Given the target eccentricity and one other orbital parameter, this function
    estimates the initial separation 'D', orbital angular velocity 'Omega_0',
    and radial expansion velocity 'adot_0' of a binary system in general
    relativity from a Post-Newtonian (PN) approximation.
    The result can be used to start an eccentricity control procedure to tune
    the initial orbital parameters further.

    Currently only zero eccentricity (circular orbits) are supported, and the
    implementation uses functions from SpEC's ZeroEccParamsFromPN.py. This
    should be generalized.

    Arguments:
      mass_ratio: Defined as q = M_A / M_B >= 1.
      dimensionless_spin_a: Dimensionless spin of the larger black hole, chi_A.
      dimensionless_spin_b: Dimensionless spin of the smaller black hole, chi_B.
      eccentricity: Eccentricity of the orbit. Specify together with _one_ of
        the other orbital parameters. Currently only an eccentricity of 0 is
        supported (circular orbit).
      mean_anomaly_fraction: Mean anomaly of the orbit divided by 2 pi, so it is
        a number between 0 and 1. The value 0 corresponds to the pericenter of
        the orbit (closest approach), the value 0.5 corresponds to the apocenter
        of the orbit (farthest distance), and the value 1 corresponds to the
        pericenter again. Currently this is unused because only an eccentricity
        of 0 is supported.
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.
      num_orbits: Number of orbits until merger.
      time_to_merger: Time to merger.

    Returns: Tuple of the initial separation 'D_0', orbital angular velocity
      'Omega_0', and radial expansion velocity 'adot_0'.
    """
    dimensionless_spin_a = np.asarray(dimensionless_spin_a)
    dimensionless_spin_b = np.asarray(dimensionless_spin_b)
    if eccentricity is not None and eccentricity != 0.0:
        assert mean_anomaly_fraction is not None, (
            "If you specify a nonzero 'eccentricity' you must also specify a"
            " 'mean_anomaly_fraction'."
        )
    if eccentricity is None:
        assert (
            separation is not None
            and orbital_angular_velocity is not None
            and radial_expansion_velocity is not None
        ), (
            "Specify all orbital parameters 'separation',"
            " 'orbital_angular_velocity', and 'radial_expansion_velocity', or"
            " specify an 'eccentricity' plus one orbital parameter."
        )
        return separation, orbital_angular_velocity, radial_expansion_velocity

    # The functions from SpEC currently work only for zero eccentricity. We will
    # need to generalize this for eccentric orbits.
    assert eccentricity == 0.0, (
        "Initial orbital parameters can currently only be computed for zero"
        " eccentricity."
    )
    assert radial_expansion_velocity is None, (
        "Can't use the 'radial_expansion_velocity' to compute orbital"
        " parameters. Remove it and choose another orbital parameter."
    )
    assert (
        (separation is not None)
        ^ (orbital_angular_velocity is not None)
        ^ (num_orbits is not None)
        ^ (time_to_merger is not None)
    ), (
        "Specify an 'eccentricity' plus _one_ of the following orbital"
        " parameters: 'separation', 'orbital_angular_velocity', 'num_orbits',"
        " 'time_to_merger'."
    )

    # Import functions from SpEC until we have ported them over. These functions
    # call old Fortran code (LSODA) through scipy.integrate.odeint, which leads
    # to lots of noise in stdout. When porting these functions, we should
    # modernize them to use scipy.integrate.solve_ivp.
    try:
        from ZeroEccParamsFromPN import nOrbitsAndTotalTime, omegaAndAdot
    except ImportError:
        raise ImportError(
            "Importing from SpEC failed. Make sure you have pointed "
            "'-D SPEC_ROOT' to a SpEC installation when configuring the build "
            "with CMake."
        )

    # Find an omega0 that gives the right number of orbits or time to merger
    if num_orbits is not None or time_to_merger is not None:
        opt_result = minimize(
            lambda x: (
                abs(
                    nOrbitsAndTotalTime(
                        q=mass_ratio,
                        chiA0=dimensionless_spin_a,
                        chiB0=dimensionless_spin_b,
                        omega0=x[0],
                    )[0 if num_orbits is not None else 1]
                    - (num_orbits if num_orbits is not None else time_to_merger)
                )
            ),
            x0=[0.01],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find an orbital angular velocity that gives the"
                " desired number of orbits or time to merger. Error:"
                f" {opt_result.message}"
            )
        orbital_angular_velocity = opt_result.x[0]
        logger.debug(
            f"Found orbital angular velocity: {orbital_angular_velocity}"
        )

    # Find the separation that gives the desired orbital angular velocity
    if orbital_angular_velocity is not None:
        opt_result = minimize(
            lambda x: abs(
                omegaAndAdot(
                    r=x[0],
                    q=mass_ratio,
                    chiA=dimensionless_spin_a,
                    chiB=dimensionless_spin_b,
                    rPrime0=1.0,  # Choice also made in SpEC
                )[0]
                - orbital_angular_velocity
            ),
            x0=[10.0],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find a separation that gives the desired orbital"
                f" angular velocity. Error: {opt_result.message}"
            )
        separation = opt_result.x[0]
        logger.debug(f"Found initial separation: {separation}")

    # Find the radial expansion velocity
    new_orbital_angular_velocity, radial_expansion_velocity = omegaAndAdot(
        r=separation,
        q=mass_ratio,
        chiA=dimensionless_spin_a,
        chiB=dimensionless_spin_b,
        rPrime0=1.0,  # Choice also made in SpEC
    )
    if orbital_angular_velocity is None:
        orbital_angular_velocity = new_orbital_angular_velocity
    else:
        assert np.isclose(
            new_orbital_angular_velocity, orbital_angular_velocity, rtol=1e-4
        ), (
            "Orbital angular velocity is inconsistent with separation."
            " Maybe the rootfind failed to reach sufficient accuracy."
        )

    # Estimate number of orbits and time to merger
    num_orbits, time_to_merger = nOrbitsAndTotalTime(
        q=mass_ratio,
        chiA0=dimensionless_spin_a,
        chiB0=dimensionless_spin_b,
        omega0=orbital_angular_velocity,
    )
    logger.info(
        "Selected approximately circular orbit. Number of orbits:"
        f" {num_orbits:g}. Time to merger: {time_to_merger:g} M."
    )
    return separation, orbital_angular_velocity, radial_expansion_velocity
