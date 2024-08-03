# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spectre.Visualization.ReadH5 import to_dataframe

from .ExecutableStatus import EvolutionStatus, list_reduction_files

logger = logging.getLogger(__name__)


class EvolveGhBinaryBlackHole(EvolutionStatus):
    executable_name_patterns = [r"^EvolveGhBinaryBlackHole"]
    fields = {
        "Time": "M",
        "Speed": "M/h",
        "Orbits": None,
        "Separation": "M",
        "3-Index Constraint": None,
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
            # Number of orbits. We use the rotation control system for this.
            try:
                rotation_z = to_dataframe(
                    open_reductions_file["ControlSystems/Rotation/z.dat"],
                    slice=np.s_[-1:],
                )
                # Assume the initial rotation angle is 0 for now. We can update
                # this to read the initial rotation angle once we can read
                # previous segments / checkpoints of a simulation.
                covered_angle = rotation_z["FunctionOfTime"].iloc[-1]
                result["Orbits"] = covered_angle / (2.0 * np.pi)
            except:
                logger.debug("Unable to extract orbits.", exc_info=True)
            # Euclidean separation between horizons
            try:
                ah_centers = [
                    to_dataframe(
                        open_reductions_file[
                            f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat"
                        ],
                        slice=np.s_[-1:],
                    ).iloc[-1]
                    for ab in "AB"
                ]
                ah_separation = np.sqrt(
                    sum(
                        (
                            ah_centers[0]["InertialCenter" + xyz]
                            - ah_centers[1]["InertialCenter" + xyz]
                        )
                        ** 2
                        for xyz in ["_x", "_y", "_z"]
                    )
                )
                result["Separation"] = ah_separation
            except:
                logger.debug("Unable to extract separation.", exc_info=True)
            # Norms
            try:
                norms = to_dataframe(
                    open_reductions_file["Norms.dat"], slice=np.s_[-1:]
                )
                result["3-Index Constraint"] = norms.iloc[-1][
                    "L2Norm(PointwiseL2Norm(ThreeIndexConstraint))"
                ]
            except:
                logger.debug(
                    "Unable to extract three index constraint.", exc_info=True
                )
        return result

    def format(self, field, value):
        if field == "Separation":
            return f"{value:g}"
        elif field == "Orbits":
            return f"{value:g}"
        elif field == "3-Index Constraint":
            return f"{value:.2e}"
        return super().format(field, value)

    def render_dashboard(self, job: dict, input_file: dict):
        import matplotlib.pyplot as plt
        import plotly.express as px
        import streamlit as st

        from spectre.Pipelines.EccentricityControl.EccentricityControl import (
            coordinate_separation_eccentricity_control,
        )
        from spectre.Visualization.GenerateXdmf import generate_xdmf
        from spectre.Visualization.PlotControlSystem import plot_control_system
        from spectre.Visualization.PlotSizeControl import plot_size_control
        from spectre.Visualization.PlotTrajectories import (
            import_A_and_B,
            plot_trajectory,
        )

        run_dir = Path(job["WorkDir"])
        reduction_files = list_reduction_files(job=job, input_file=input_file)

        # Common horizon
        with h5py.File(run_dir / reduction_files[-1], "r") as open_h5file:
            ahc_subfile = open_h5file.get("ObservationAhC.dat")
            if ahc_subfile is not None:
                ahc_data = to_dataframe(ahc_subfile).iloc[0]
                st.success(
                    "Found a common horizon with mass"
                    f" {ahc_data['ChristodoulouMass']:g} and spin"
                    f" {ahc_data['DimensionlessSpinMagnitude']:g} at time"
                    f" {ahc_data['Time']:g} M."
                )

        # Plot style
        legend_layout = dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0,
        )

        # Constraints
        st.subheader("Constraints")

        def get_constraints_data(reductions_file):
            with h5py.File(reductions_file, "r") as open_h5file:
                return to_dataframe(open_h5file["Norms.dat"]).set_index("Time")

        constraints = pd.concat(map(get_constraints_data, reduction_files))
        constraints.sort_index(inplace=True)
        constraints = constraints[
            [col for col in constraints.columns if "Constraint" in col]
        ]
        fig = px.line(constraints.iloc[1:], log_y=True)
        fig.update_layout(xaxis_title="Time [M]", legend=legend_layout)
        fig.update_yaxes(exponentformat="e", title=None)
        st.plotly_chart(fig)

        # Horizon parameters
        st.subheader("Horizon parameters")

        def get_horizons_data(reductions_file):
            with h5py.File(reductions_file, "r") as open_h5file:
                horizons_data = []
                for ab in "AB":
                    ah_subfile = open_h5file.get(f"ObservationAh{ab}.dat")
                    if ah_subfile is not None:
                        horizons_data.append(
                            to_dataframe(ah_subfile)
                            .set_index("Time")
                            .add_prefix(f"Ah{ab} ")
                        )
                if not horizons_data:
                    return None
                return pd.concat(horizons_data, axis=1)

        horizon_params = pd.concat(map(get_horizons_data, reduction_files))
        for label, name, col in zip(
            ["AhA mass", "AhB mass", "AhA spin", "AhB spin"],
            [
                "AhA ChristodoulouMass",
                "AhB ChristodoulouMass",
                "AhA DimensionlessSpinMagnitude",
                "AhB DimensionlessSpinMagnitude",
            ],
            st.columns(4),
        ):
            col.metric(label, f"{horizon_params.iloc[-1][name]:.4g}")
        fig = px.line(
            np.abs(horizon_params.iloc[1:] - horizon_params.iloc[0]),
            log_y=True,
        )
        fig.update_layout(xaxis_title="Time [M]", legend=legend_layout)
        fig.update_yaxes(
            exponentformat="e", title="Difference to initial values"
        )
        st.plotly_chart(fig)

        # Time steps
        super().render_time_steps(input_file, reduction_files)

        # Trajectories
        st.subheader("Trajectories")
        traj_A, traj_B = import_A_and_B(reduction_files)
        st.pyplot(plot_trajectory(traj_A, traj_B))
        coord_separation = np.linalg.norm(traj_A[:, 1:] - traj_B[:, 1:], axis=1)
        fig = px.line(
            x=traj_A[:, 0],
            y=coord_separation,
            labels={"y": "Coordinate separation [M]"},
        )
        fig.update_layout(xaxis_title="Time [M]", legend=legend_layout)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig)

        # Grid
        st.subheader("Grid")

        @st.fragment
        def render_grid():
            if st.button("Render grid"):
                from spectre.Visualization.Render3D.Domain import render_domain

                volume_files = glob.glob(str(run_dir / "BbhVolume*.h5"))
                generate_xdmf(
                    volume_files,
                    output=str(run_dir / "Bbh.xmf"),
                    subfile_name="VolumeData",
                )
                render_domain(
                    str(run_dir / "Bbh.xmf"),
                    output=str(run_dir / "domain.png"),
                    slice=True,
                    zoom_factor=50.0,
                    time_step=-1,
                )
            if (run_dir / "domain.png").exists():
                st.image(str(run_dir / "domain.png"))

        render_grid()

        # Control systems
        st.subheader("Control systems")

        @st.fragment
        def render_control_systems():
            if st.checkbox("Show control systems"):
                st.pyplot(
                    plot_control_system(
                        reduction_files,
                        with_shape=st.checkbox("With shape", True),
                        show_all_m=st.checkbox("Show all m", False),
                        shape_l_max=st.number_input(
                            "Shape l max", value=2, min_value=0
                        ),
                    )
                )
                with st.expander("Size control A", expanded=False):
                    st.pyplot(plot_size_control(reduction_files, "A"))
                with st.expander("Size control B", expanded=False):
                    st.pyplot(plot_size_control(reduction_files, "B"))

        render_control_systems()

        # Eccentricity
        st.subheader("Eccentricity")

        @st.fragment
        def render_eccentricity():
            if st.checkbox("Show eccentricity"):
                col_tmin, col_tmax = st.columns(2)
                fig = plt.figure(figsize=(10, 8), layout="tight")
                ecc_control_result = coordinate_separation_eccentricity_control(
                    reduction_files[0],
                    "ApparentHorizons/ControlSystemAhA_Centers.dat",
                    "ApparentHorizons/ControlSystemAhB_Centers.dat",
                    tmin=col_tmin.number_input("tmin", value=600, min_value=0),
                    tmax=col_tmax.number_input("tmax", value=2000, min_value=0),
                    angular_velocity_from_xcts=None,
                    expansion_from_xcts=None,
                    fig=fig,
                )["H4"]["fit result"]
                st.pyplot(fig)
                st.metric(
                    "Eccentricity",
                    f"{ecc_control_result['eccentricity']:e}",
                )
                st.write(ecc_control_result)

        render_eccentricity()
