# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.Evolution.Ringdown as Ringdown
import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import ModalVector
from spectre.SphericalHarmonics import Strahlkorper, ylm_legend_and_data

logger = logging.getLogger(__name__)


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def dt_cubic(x, a, b, c, d):
    return 3 * a * x**2 + 2 * b * x + c


def dt2_cubic(x, a, b, c, d):
    return 6 * a * x + 2 * b


# Cubic fit transformed coefs to get first and second time derivatives
def fit_to_a_cubic(times, coefs, match_time, zero_coefs_eps):
    fits = []
    fit_ahc = []
    fit_dt_ahc = []
    fit_dt2_ahc = []
    for j in np.arange(0, coefs.shape[-1], 1):
        # Optionally, avoid fitting coefficients sufficiently close to zero by
        # just setting these coefficients and their time derivatives to zero.
        if (
            zero_coefs_eps is not None
            and sum(np.abs(coefs[:, j])) < zero_coefs_eps
        ):
            fits.append(np.zeros(4))
            fit_ahc.append(0.0)
            fit_dt_ahc.append(0.0)
            fit_dt2_ahc.append(0.0)
            continue
        # Ignore RankWarnings suggesting the fit might not be good enough;
        # for equal-mass non-spinning, sufficiently good fits for starting
        # a ringdown, even though RankWarnings were triggered
        with warnings.catch_warnings():
            # In numpy 2.0+, RankWarning was moved to np.exceptions.RankWarning
            try:
                warnings.simplefilter("ignore", np.RankWarning)
            except AttributeError:
                warnings.simplefilter("ignore", np.exceptions.RankWarning)
            fit = np.polyfit(times, coefs[:, j], 3)
        fits.append(fit)
        fit_ahc.append(cubic(match_time, *(fit)))
        fit_dt_ahc.append(dt_cubic(match_time, *(fit)))
        fit_dt2_ahc.append(dt2_cubic(match_time, *(fit)))

    return fit_ahc, fit_dt_ahc, fit_dt2_ahc


def compute_ahc_coefs_in_ringdown_distorted_frame(
    ahc_reductions_path,
    ahc_subfile,
    exp_func_and_2_derivs,
    exp_outer_bdry_func_and_2_derivs,
    rot_func_and_2_derivs,
    number_of_ahc_finds_for_fit,
    match_time,
    settling_timescale,
    zero_coefs_eps,
):
    """Computes the AhC Ylm Coefficients in the Ringdown distorted frame
    using the functions of time and the AhC coefficients in the Inspiral
    inertial frame.

    Arugments:
    ahc_reductions_path: Path to reduction file where AhC Coefficients will be
    written.
    ahc_subfile: The subfile of the reductions file where AhC coefficients will
    be placed.
    expansion_func_and_2_derivs: Expansion functions of time from volume
    data.
    exp_outer_bdry_func_and_2_derivs: Outer boundary expansion function of time
    from volume data
    rot_func_and_2_derivs: Rotation function of time from volume data
    number_of_ahc_finds_for_fit: The number of ahc finds that will be used in
    the fit.
    match_time: Time to match functions of time.
    settling_timescale: Timescale for settle to constant functions of time.
    zero_coefs_eps: Approximate limit to compare coefficients to 0.0

    """

    ahc_times = []
    ahc_center = []
    ahc_lmax = 0
    with spectre_h5.H5File(ahc_reductions_path, "r") as h5file:
        datfile = h5file.get_dat(ahc_subfile)
        datfile_np = np.array(datfile.get_data())
        ahc_times = datfile_np[:, 0]
        ahc_center = [datfile_np[0][1], datfile_np[0][2], datfile_np[0][3]]
        ahc_lmax = int(datfile_np[0][4])

    # Transform AhC coefs to ringdown distorted frame and get other data
    # needed to start a ringdown, such as initial values for functions of time
    coefs_at_different_times = np.array(
        Ringdown.strahlkorper_coefs_in_ringdown_distorted_frame(
            ahc_reductions_path,
            ahc_subfile,
            number_of_ahc_finds_for_fit,
            match_time,
            settling_timescale,
            exp_func_and_2_derivs,
            exp_outer_bdry_func_and_2_derivs,
            rot_func_and_2_derivs,
        )
    )

    # Do not include AhCs at times greater than the match time. Errors tend
    # to grow as time increases, so fit derivatives using the match time
    # and earlier times, to get a more accurate fit.
    ahc_times_for_fit_list = []
    coefs_at_different_times_for_fit_list = []
    for i, time in enumerate(ahc_times[-number_of_ahc_finds_for_fit:]):
        if time <= match_time:
            ahc_times_for_fit_list.append(time)
            coefs_at_different_times_for_fit_list.append(
                coefs_at_different_times[i]
            )
    ahc_times_for_fit = np.array(ahc_times_for_fit_list)
    coefs_at_different_times_for_fit = np.array(
        coefs_at_different_times_for_fit_list
    )
    if ahc_times_for_fit.shape[0] == 0:
        logger.warning(
            "No available AhC times before selected match time; using all"
            " available AhC times, even though numerical errors are likely"
            " larger after the match time"
        )
        ahc_times_for_fit = ahc_times[-number_of_ahc_finds_for_fit:]
        coefs_at_different_times_for_fit = coefs_at_different_times[
            -number_of_ahc_finds_for_fit:
        ]

    logger.info("AhC times available: " + str(ahc_times.shape[0]))
    logger.info(
        "AhC available time range: "
        + str(np.min(ahc_times))
        + " - "
        + str(np.max(ahc_times))
    )
    logger.info("AhC times used: " + str(ahc_times_for_fit.shape[0]))
    logger.info(
        "AhC used time range: "
        + str(np.min(ahc_times_for_fit))
        + " - "
        + str(np.max(ahc_times_for_fit))
    )
    logger.info(
        "Coef times available: " + str(coefs_at_different_times.shape[0])
    )
    logger.info(
        "Coef times used: " + str(coefs_at_different_times_for_fit.shape[0])
    )

    fit_ahc_coefs, fit_ahc_dt_coefs, fit_ahc_dt2_coefs = fit_to_a_cubic(
        ahc_times_for_fit,
        coefs_at_different_times_for_fit,
        match_time,
        zero_coefs_eps,
    )

    # Note: assumes no translation, so inertial and distorted centers are the
    # same, i.e. both are at the origin. A future update will incorporate
    # translation.

    fit_ahc_coef_mv = ModalVector(fit_ahc_coefs)
    fit_ahc_dt_coef_mv = ModalVector(fit_ahc_dt_coefs)
    fit_ahc_dt2_coef_mv = ModalVector(fit_ahc_dt2_coefs)
    fit_ahc_strahlkorper = Strahlkorper(
        ahc_lmax, ahc_lmax, fit_ahc_coef_mv, ahc_center
    )
    fit_ahc_dt_strahlkorper = Strahlkorper(
        ahc_lmax, ahc_lmax, fit_ahc_dt_coef_mv, ahc_center
    )
    fit_ahc_dt2_strahlkorper = Strahlkorper(
        ahc_lmax, ahc_lmax, fit_ahc_dt2_coef_mv, ahc_center
    )
    legend_ahc, fit_ahc_ylm_coefs_to_write = ylm_legend_and_data(
        fit_ahc_strahlkorper, match_time, ahc_lmax
    )
    legend_ahc_dt, fit_ahc_dt_ylm_coefs_to_write = ylm_legend_and_data(
        fit_ahc_dt_strahlkorper, match_time, ahc_lmax
    )
    legend_ahc_dt2, fit_ahc_dt2_ylm_coefs_to_write = ylm_legend_and_data(
        fit_ahc_dt2_strahlkorper, match_time, ahc_lmax
    )

    ringdown_ylm_coefs = [
        fit_ahc_ylm_coefs_to_write,
        fit_ahc_dt_ylm_coefs_to_write,
        fit_ahc_dt2_ylm_coefs_to_write,
    ]
    ringdown_ylm_legend = [legend_ahc, legend_ahc_dt, legend_ahc_dt2]

    for i in range(1, 4, 1):
        legend_ahc[i] = legend_ahc[i].replace("Inertial", "Distorted")
        legend_ahc_dt[i] = legend_ahc_dt[i].replace("Inertial", "Distorted")
        legend_ahc_dt2[i] = legend_ahc_dt2[i].replace("Inertial", "Distorted")

    return ringdown_ylm_coefs, ringdown_ylm_legend
