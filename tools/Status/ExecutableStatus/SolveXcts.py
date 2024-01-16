# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import h5py

from .ExecutableStatus import EllipticStatus

logger = logging.getLogger(__name__)


class SolveXcts(EllipticStatus):
    executable_name_patterns = [r"^SolveXcts"]
    fields = {
        "Nonlinear iteration": None,
        "Nonlinear residual": None,
        "Linear iteration": None,
        "Linear residual": None,
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
            nonlinear_status = self.solver_status(
                input_file, open_reductions_file, solver_name="NewtonRaphson"
            )
            linear_status = self.solver_status(
                input_file, open_reductions_file, solver_name="Gmres"
            )
        result = {
            "Nonlinear iteration": nonlinear_status["Iteration"],
            "Nonlinear residual": nonlinear_status["Residual"],
            "Linear iteration": linear_status["Iteration"],
            "Linear residual": linear_status["Residual"],
        }
        return result

    def format(self, field, value):
        if "iteration" in field:
            return super().format("Iteration", value)
        elif "residual" in field:
            return super().format("Residual", value)
