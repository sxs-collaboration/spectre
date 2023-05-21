# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import h5py
import numpy as np

from spectre.Visualization.ReadH5 import to_dataframe

from .ExecutableStatus import EvolutionStatus

logger = logging.getLogger(__name__)


class EvolveGhSingleBlackHole(EvolutionStatus):
    executable_name_patterns = [r"^EvolveGhSingleBlackHole"]
    fields = {
        "Time": "M",
        "Speed": "M/h",
        "Constraint Energy": None,
    }

    def status(self, input_file, work_dir):
        try:
            reductions_file = input_file["Observers"]["ReductionFileName"]
            open_reductions_file = h5py.File(
                os.path.join(work_dir, reductions_file + ".h5"), "r"
            )
        except:
            logger.debug("Unable to open reductions file.", exc_info=True)
            return {}
        with open_reductions_file:
            result = self.time_status(input_file, open_reductions_file)
            # Norms
            try:
                norms = to_dataframe(
                    open_reductions_file["Norms.dat"], slice=np.s_[-1:]
                )
                result["Constraint Energy"] = norms.iloc[-1][
                    "L2Norm(ConstraintEnergy)"
                ]
            except:
                logger.debug(
                    "Unable to extract constraint energy.", exc_info=True
                )
        return result

    def format(self, field, value):
        if field in ["Constraint Energy"]:
            return f"{value:.2e}"
        return super().format(field, value)
