# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    def render_dashboard(self, job: dict, input_file: dict):
        import plotly.express as px
        import streamlit as st

        super().render_solver_convergence(job, input_file)

        # Parameter control
        control_outfile = Path(job["WorkDir"]) / "ControlParamsData.txt"
        if control_outfile.exists():
            st.subheader("Parameter control")
            control_data = np.loadtxt(control_outfile, delimiter=",")
            df = pd.DataFrame(
                control_data,
                columns=[
                    "Iteration",
                    "Mass A",
                    "Mass B",
                    "Spin A x",
                    "Spin A y",
                    "Spin A z",
                    "Spin B x",
                    "Spin B y",
                    "Spin B z",
                    "Residual Mass A",
                    "Residual Mass B",
                    "Residual Spin A x",
                    "Residual Spin A y",
                    "Residual Spin A z",
                    "Residual Spin B x",
                    "Residual Spin B y",
                    "Residual Spin B z",
                ],
            ).set_index("Iteration")
            # Print current params
            params = df.iloc[-1]
            mass_A, mass_B = params["Mass A"], params["Mass B"]
            spin_A, spin_B = [
                np.linalg.norm(params[[f"Spin {AB} {xyz}" for xyz in "xyz"]])
                for AB in "AB"
            ]
            for label, val, col in zip(
                ["Mass A", "Mass B", "Spin A", "Spin B"],
                [mass_A, mass_B, spin_A, spin_B],
                st.columns(4),
            ):
                col.metric(label, f"{val:.4g}")
            # Plot convergence
            fig = px.line(
                np.abs(df[[col for col in df.columns if "Residual" in col]]),
                log_y=True,
                markers=True,
            )
            fig.update_yaxes(exponentformat="e", title=None)
            st.plotly_chart(fig)
