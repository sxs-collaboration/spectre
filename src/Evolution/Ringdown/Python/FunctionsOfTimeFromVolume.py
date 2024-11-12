# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.IO.H5 as spectre_h5
from spectre.Domain import deserialize_functions_of_time

logger = logging.getLogger(__name__)


# Transform AhC coefs to ringdown distorted frame and get other data
# needed to start a ringdown, such as initial values for functions of time
def functions_of_time_from_volume(
    fot_vol_h5_path, fot_vol_subfile, match_time, which_obs_id
):
    exp_func_and_2_derivs = []
    exp_outer_bdry_func_and_2_derivs = []
    rot_func_and_2_derivs = []

    with spectre_h5.H5File(fot_vol_h5_path, "r") as h5file:
        if fot_vol_subfile.split(".")[-1] == "vol":
            fot_vol_subfile = fot_vol_subfile.split(".")[0]
        volfile = h5file.get_vol("/" + fot_vol_subfile)
        obs_ids = volfile.list_observation_ids()
        logger.info("About to deserialize functions of time")
        fot_times = list(map(volfile.get_observation_value, obs_ids))
        serialized_fots = volfile.get_functions_of_time(obs_ids[which_obs_id])
        functions_of_time = deserialize_functions_of_time(serialized_fots)
        logger.info("Deserialized functions of time")

        # The inspiral expansion map includes two functions of time:
        # an expansion map allowing the black holes to move closer together in
        # comoving coordinates, and an expansion map causing the outer boundary
        # to move slightly inwards. The ringdown only requires the outer
        # boundary expansion map, so set the other map to the identity.
        exp_func_and_2_derivs = [1.0, 0.0, 0.0]

        exp_outer_bdry_func_and_2_derivs = [
            x[0]
            for x in functions_of_time[
                "ExpansionOuterBoundary"
            ].func_and_2_derivs(fot_times[which_obs_id])
        ]
        rot_func_and_2_derivs_tuple = functions_of_time[
            "Rotation"
        ].func_and_2_derivs(fot_times[which_obs_id])
        rot_func_and_2_derivs = [
            [coef for coef in x] for x in rot_func_and_2_derivs_tuple
        ]

    return (
        exp_func_and_2_derivs,
        exp_outer_bdry_func_and_2_derivs,
        rot_func_and_2_derivs,
    )
