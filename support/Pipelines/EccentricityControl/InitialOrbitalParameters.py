# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# The following two functions are modernized versions of those in SpEC's
# `ZeroEccParamsFromPN.py`. They use higher PN orders (whichever are implemented
# in the PostNewtonian module), are much faster, and avoid spurious output from
# old Fortran code (LSODA) that was used in SpEC's `ZeroEccParamsFromPN.py`.
# They are consistent with SpEC up to 2.5 PN order, as tested by Mike Boyle (see
# https://github.com/moble/PostNewtonian.jl/issues/41).
#
# Since these functions use Julia through Python bindings, they will download
# Julia and precompile the packages on first use, which may take a few minutes
# (see https://moble.github.io/PostNewtonian.jl/dev/interface/python/).


def omega_and_adot(r, q, chiA, chiB):
    from sxs.julia import PostNewtonian

    pn = PostNewtonian.BBH(
        np.array(
            [1.0 / (1.0 + q), q / (1.0 + q), *chiA, *chiB, 1, 0, 0, 0, 1, 0]
        )
    )
    pn.state[12] = PostNewtonian.separation_inverse(r, pn)
    return PostNewtonian.Omega(pn), PostNewtonian.separation_dot(pn) / r


def num_orbits_and_time_to_merger(q, chiA0, chiB0, omega0):
    from sxs.julia import PNWaveform

    pn_waveform = PNWaveform(
        M1=1.0 / (1.0 + q),
        M2=q / (1.0 + q),
        chi1=chiA0,
        chi2=chiB0,
        Omega_i=omega0,
    )
    return 0.5 * pn_waveform.orbital_phase[-1] / np.pi, pn_waveform.time[-1]


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

    # The functions from the PostNewtonian module currently work only for zero
    # eccentricity. We will need to generalize this for eccentric orbits.
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

    # Find an omega0 that gives the right number of orbits or time to merger
    if num_orbits is not None or time_to_merger is not None:
        logger.info("Finding orbital angular velocity...")
        opt_result = minimize(
            lambda x: (
                abs(
                    num_orbits_and_time_to_merger(
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
        logger.info("Finding separation...")
        opt_result = minimize(
            lambda x: abs(
                omega_and_adot(
                    r=x[0],
                    q=mass_ratio,
                    chiA=dimensionless_spin_a,
                    chiB=dimensionless_spin_b,
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
    new_orbital_angular_velocity, radial_expansion_velocity = omega_and_adot(
        r=separation,
        q=mass_ratio,
        chiA=dimensionless_spin_a,
        chiB=dimensionless_spin_b,
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
    num_orbits, time_to_merger = num_orbits_and_time_to_merger(
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
