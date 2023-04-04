# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import h5py
import numpy as np
from spectre.Visualization.ReadH5 import to_dataframe

from .ExecutableStatus import EvolutionStatus

logger = logging.getLogger(__name__)


class EvolveGhBinaryBlackHole(EvolutionStatus):
    executable_name_patterns = [r"^EvolveGhBinaryBlackHole"]
    fields = {
        "Time": "M",
        "Speed": "M/h",
        "Orbits": None,
        "Separation": "M",
        "Constraint Energy": None,
    }

    def status(self, input_file, work_dir):
        try:
            reductions_file = input_file["Observers"]["ReductionFileName"]
            open_reductions_file = h5py.File(
                os.path.join(work_dir, reductions_file + ".h5"), "r")
        except:
            logger.debug("Unable to open reductions file.", exc_info=True)
            return {}
        with open_reductions_file:
            result = self.time_status(input_file, open_reductions_file)
            # Number of orbits. We use the rotation control system for this.
            try:
                rotation_z = to_dataframe(
                    open_reductions_file["ControlSystems/Rotation/z.dat"])
                covered_angle = np.diff(rotation_z["Lambda"].iloc[[0, -1]])[0]
                result["Orbits"] = covered_angle / (2. * np.pi)
            except:
                logger.debug("Unable to extract orbits.", exc_info=True)
            # Euclidean separation between horizons
            try:
                ah_centers = [
                    to_dataframe(open_reductions_file[
                        f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat"]).
                    iloc[-1] for ab in "AB"
                ]
                ah_separation = np.sqrt(
                    sum((ah_centers[0]["InertialCenter" + xyz] -
                         ah_centers[1]["InertialCenter" + xyz])**2
                        for xyz in ["_x", "_y", "_z"]))
                result["Separation"] = ah_separation
            except:
                logger.debug("Unable to extract separation.", exc_info=True)
            # Norms
            try:
                norms = to_dataframe(open_reductions_file["Norms.dat"])
                result["Constraint Energy"] = norms.iloc[-1][
                    "L2Norm(ConstraintEnergy)"]
            except:
                logger.debug("Unable to extract constraint energy.",
                             exc_info=True)
        return result

    def format(self, field, value):
        if field == "Separation":
            return f"{value:g}"
        elif field == "Orbits":
            return f"{value:g}"
        elif field == "Constraint Energy":
            return f"{value:.2e}"
        return super().format(field, value)
