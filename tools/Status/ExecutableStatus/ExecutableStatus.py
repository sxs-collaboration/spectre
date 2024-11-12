# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectre.support.DirectoryStructure import list_segments
from spectre.Visualization.ReadH5 import to_dataframe
from spectre.Visualization.ReadInputFile import find_event

logger = logging.getLogger(__name__)


def list_reduction_files(job: dict, input_file: dict):
    reductions_file_name = input_file["Observers"]["ReductionFileName"] + ".h5"
    if job["SegmentsDir"]:
        return [
            segment_dir.path / reductions_file_name
            for segment_dir in list_segments(job["SegmentsDir"])
        ]
    else:
        return [Path(job["WorkDir"]) / reductions_file_name]


class ExecutableStatus:
    """Base class for providing status information for SpECTRE executables.

    Subclasses read output from executables and provide status information.
    For example, a subclass for a binary black hole evolution executable can
    provide the current separation and number of orbits.

    Properties:
      executable_name_patterns: List of regex patterns that match the executable
        names the subclass applies to. For example, r"^Evolve" matches all
        executables that start with "Evolve".
      fields: Dictionary of all field names the subclass provides and their
        units. For example: {"Time": "M", "Orbits": None} if the subclass
        provides time in units of mass and orbits with no unit.
    """

    executable_name_patterns: List[str] = []
    fields: Dict[str, Optional[str]] = {}

    def status(
        self, input_file: Optional[dict], work_dir: Optional[str]
    ) -> dict:
        """Provide status information of an executable run.

        Arguments:
          input_file: The input file read in as a dictionary.
          work_dir: The working directory of the executable run. Paths in the
            input file are relative to this directory.

        Returns: Dictionary of 'fields' and their values. For example:
          {"Time": 10.5, "Orbits": 3}. Not all fields must be provided if they
          can't be determined.

        Note: Avoid raising exceptions in this function. It is better to provide
          only a subset of fields than to raise an exception. For example, if
          there's an error reading the number of orbits, intercept and log the
          exception and just skip providing the number of orbits.
        """
        return {}

    def format(self, field: str, value: Any) -> str:
        """Transform status values to human-readable strings.

        Arguments:
          field: The name of the field, as listed in 'fields' (e.g. "Time").
          value: The status value of the field, as returned by 'status()'.

        Returns: Human-readable representation of the value.
        """
        raise NotImplementedError

    def render_dashboard(self, job: dict, input_file: dict):
        """Render a dashboard for the job (experimental).

        Arguments:
          job: A dictionary of job information, including the input file and
            working directory. See the 'spectre.tools.Status.fetch_status'
            function for details.
          input_file: The input file read in as a dictionary.

        This method can be overridden by subclasses to provide a custom
        dashboard for the job. The default implementation does nothing.
        """
        import streamlit as st

        st.warning(
            "No dashboard available for this executable. Add an implementation"
            " to the 'ExecutableStatus' subclass."
        )


class EvolutionStatus(ExecutableStatus):
    """An 'ExecutableStatus' subclass that matches all evolution executables.

    This is a fallback if no more specialized subclass is implemented. It just
    determines the current time and run speed. This class can be subclassed
    further to use the 'time_status' function in subclasses.
    """

    executable_name_patterns = [r"^Evolve"]
    fields = {
        "Time": None,
        "Speed": "1/h",
    }

    def time_status(
        self, input_file: dict, open_reductions_file, avg_num_slabs: int = 50
    ) -> dict:
        """Report the simulation time and speed.

        Uses the 'ObserveTimeStep' event, so the status information is only as
        current as the output frequency of the event. The reported time is the
        most recent output of the 'ObserveTimeStep' event, and the reported
        speed is determined from the two most recent outputs. Since the walltime
        at which different elements in the computational domain reach the
        timestep varies because of the lack of a global synchronization point in
        the task-based parallel evolution, this function returns the average
        between the minimum and maximum walltime.

        Arguments:
          input_file: The input file read in as a dictionary.
          open_reductions_file: The open h5py reductions data file.
          avg_num_slabs: Number of slabs to average over for estimating the
            simulation speed (at least 2).

        Returns: Status fields "Time" in code units and "Speed" in simulation
          time per hour.
        """
        assert (
            avg_num_slabs >= 2
        ), "Need at least 'avg_num_slabs >= 2' to estimate simulation speed."
        observe_time_event = find_event(
            "ObserveTimeStep", "EventsAndTriggers", input_file
        )
        if observe_time_event is None:
            return {}
        subfile_name = observe_time_event["SubfileName"] + ".dat"
        logger.debug(f"Reading time steps from subfile: '{subfile_name}'")
        try:
            # Time steps are sampled at most once per slab, so we load data
            # going back `avg_num_slabs`
            time_steps = to_dataframe(
                open_reductions_file[subfile_name], slice=np.s_[-avg_num_slabs:]
            )
        except:
            logger.debug("Unable to read time steps.", exc_info=True)
            return {}
        result = {
            "Time": time_steps.iloc[-1]["Time"],
        }
        # Estimate simulation speed
        try:
            # Average over the last `avg_num_slabs` slabs, but at least the last
            # two observations
            start_time = min(
                time_steps.iloc[-1]["Time"]
                - time_steps.iloc[-1]["Slab size"] * (avg_num_slabs - 1),
                time_steps.iloc[-2]["Time"],
            )
            time_window = time_steps[time_steps["Time"] >= start_time]
            dt_sim = time_window.iloc[-1]["Time"] - time_window.iloc[0]["Time"]
            dt_wall_min = (
                time_window.iloc[-1]["Minimum Walltime"]
                - time_window.iloc[0]["Minimum Walltime"]
            )
            dt_wall_max = (
                time_window.iloc[-1]["Maximum Walltime"]
                - time_window.iloc[0]["Maximum Walltime"]
            )
            speed = (
                (dt_sim / dt_wall_min + dt_sim / dt_wall_max)
                / 2.0
                * 60.0
                * 60.0
            )
            result["Speed"] = speed
        except:
            logger.debug("Unable to estimate simulation speed.", exc_info=True)
        return result

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
            return self.time_status(input_file, open_reductions_file)

    def format(self, field, value):
        if field in ["Time", "Speed"]:
            return f"{value:g}"
        raise ValueError

    def render_time_steps(self, input_file: dict, reduction_files: list):
        import plotly.express as px
        import streamlit as st

        observe_time_event = find_event(
            "ObserveTimeStep", "EventsAndTriggers", input_file
        )
        if observe_time_event is None:
            st.warning("No 'ObserveTimeStep' event found in input file.")
            return
        subfile_name = observe_time_event["SubfileName"] + ".dat"
        logger.debug(f"Reading time steps from subfile: '{subfile_name}'")

        def get_time_steps(reductions_file):
            with h5py.File(reductions_file, "r") as open_h5file:
                return to_dataframe(open_h5file[subfile_name]).set_index("Time")

        # Plot style
        legend_layout = dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0,
        )

        st.subheader("Time steps")
        time_steps = pd.concat(map(get_time_steps, reduction_files))
        fig = px.line(
            time_steps,
            y=[
                "Effective time step",
                "Minimum time step",
                "Maximum time step",
                "Slab size",
            ],
            log_y=True,
        )
        fig.update_layout(
            legend=legend_layout,
            xaxis=dict(title="Time [M]"),
            yaxis=dict(exponentformat="e", title=None),
        )
        st.plotly_chart(fig)
        run_speed = (
            (
                time_steps.index.diff() / time_steps["Maximum Walltime"].diff()
                + time_steps.index.diff()
                / time_steps["Minimum Walltime"].diff()
            )
            / 2
            * 3600
        )
        fig = px.line(run_speed.rolling(30, min_periods=1).mean())
        fig.update_layout(
            xaxis_title="Time [M]",
            yaxis_title="Simulation speed [M/h]",
            showlegend=False,
        )
        st.plotly_chart(fig)

    def render_dashboard(self, job: dict, input_file: dict):
        return self.render_time_steps(
            input_file, list_reduction_files(job, input_file)
        )


class EllipticStatus(ExecutableStatus):
    """An 'ExecutableStatus' subclass that matches all elliptic executables.

    This is a fallback if no more specialized subclass is implemented.
    """

    executable_name_patterns = [r"^Solve"]
    fields = {
        "Iteration": None,
        "Residual": None,
    }

    def solver_status(
        self, input_file: dict, open_reductions_file, solver_name: str
    ) -> dict:
        """Report the residual of the iterative solver

        Arguments:
          input_file: The input file read in as a dictionary.
          open_reductions_file: The open h5py reductions data file.
          solver_name: Name of the solver. Will read the subfile named
            <SOLVER_NAME>Residuals.dat from the reductions file to extract
            number of iterations and residual.

        Returns: Status fields "Iteration" and "Residual".
        """
        subfile_name = solver_name + "Residuals.dat"
        logger.debug(f"Reading residuals from subfile: '{subfile_name}'")
        try:
            residuals = to_dataframe(open_reductions_file[subfile_name])
        except:
            logger.debug("Unable to read residuals.", exc_info=True)
            return {}
        result = {
            "Iteration": len(residuals[residuals["Iteration"] != 0]),
            "Residual": residuals["Residual"].iloc[-1],
        }
        return result

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
            # Try these solver names in turn and report the status of the first
            # that exists. Create a subclass to customize this behavior for
            # a specific executable (e.g. if it has multiple solvers).
            for solver_name in ["NewtonRaphson", "Gmres"]:
                solver_status = self.solver_status(
                    input_file, open_reductions_file, solver_name=solver_name
                )
                if solver_status:
                    return solver_status
        return {}

    def format(self, field, value):
        if field in ["Iteration"]:
            return str(int(value))
        elif field in ["Residual"]:
            return f"{value:.1e}"
        raise ValueError

    def render_solver_convergence(self, job: dict, input_file: dict):
        import plotly.express as px
        import streamlit as st

        from spectre.Visualization.PlotEllipticConvergence import (
            plot_elliptic_convergence,
        )

        st.subheader("Elliptic solver convergence")
        fig = plt.figure(figsize=(8, 6), layout="tight")
        plot_elliptic_convergence(
            Path(job["WorkDir"])
            / (input_file["Observers"]["ReductionFileName"] + ".h5"),
            fig=fig,
        )
        st.pyplot(fig)

    def render_dashboard(self, job: dict, input_file: dict):
        return self.render_solver_convergence(job, input_file)
