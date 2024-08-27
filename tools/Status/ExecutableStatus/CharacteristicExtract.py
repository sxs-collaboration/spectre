# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectre.Visualization.ReadH5 import to_dataframe

from .ExecutableStatus import ExecutableStatus

logger = logging.getLogger(__name__)


class CharacteristicExtract(ExecutableStatus):
    executable_name_patterns = [r"^CharacteristicExtract"]
    fields = {
        "Time": "M",
    }

    def status(self, input_file, work_dir):
        try:
            open_reductions_file = h5py.File(
                Path(work_dir)
                / (input_file["Observers"]["ReductionFileName"] + ".h5"),
                "r",
            )
        except:
            logger.debug("Unable to open reductions file.", exc_info=True)
            return {}
        with open_reductions_file:
            try:
                time_steps = to_dataframe(
                    open_reductions_file["Cce/CceTimeStep.dat"]
                )
            except:
                logger.debug("Unable to read CCE time steps.", exc_info=True)
                return {}
        return {
            "Time": time_steps.iloc[-1]["Time"],
        }

    def format(self, field, value):
        if field in ["Time", "Speed"]:
            return f"{value:g}"
        raise ValueError

    def render_dashboard(self, job: dict, input_file: dict):
        import plotly.express as px
        import streamlit as st

        from spectre.Visualization.PlotCce import plot_cce

        fig = plot_cce(
            Path(job["WorkDir"])
            / (input_file["Observers"]["ReductionFileName"] + ".h5"),
            modes=["Real Y_2,2", "Imag Y_2,2"],
            x_label="Time [M]",
        )
        st.pyplot(fig)
