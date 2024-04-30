# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
from pathlib import Path
from typing import Optional, Sequence, Type, Union

import click

import spectre.IO.H5 as spectre_h5
from spectre.ApparentHorizonFinder import FastFlow, FlowType, Status
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr
from spectre.IO.Exporter import interpolate_tensors_to_points
from spectre.Spectral import Basis, Quadrature
from spectre.SphericalHarmonics import (
    Strahlkorper,
    cartesian_coords,
    ylm_legend_and_data,
)
from spectre.Visualization.OpenVolfiles import open_volfiles_command
from spectre.Visualization.ReadH5 import list_observations

logger = logging.getLogger(__name__)


def _strahlkorper_vol_data(strahlkorper):
    """Volume data representation of a Strahlkorper, can be written to H5."""
    coords = cartesian_coords(strahlkorper)
    return [
        spectre_h5.ElementVolumeData(
            element_name="Horizon",
            components=[
                spectre_h5.TensorComponent(
                    "InertialCoordinates_" + "xyz"[d], coords[d]
                )
                for d in range(3)
            ],
            extents=strahlkorper.physical_extents,
            basis=2 * [Basis.SphericalHarmonic],
            quadrature=[
                Quadrature.Gauss,
                Quadrature.Equiangular,
            ],
        )
    ]


def find_horizon(
    h5_files: Union[str, Sequence[str]],
    subfile_name: str,
    obs_id: int,
    obs_time: float,
    initial_guess: Strahlkorper,
    fast_flow: Optional[FastFlow] = None,
    output: Optional[str] = None,
    output_coeffs_subfile: Optional[str] = None,
    output_coords_subfile: Optional[str] = None,
    output_l_max: Optional[int] = None,
    tensor_names: Optional[Sequence[str]] = None,
):
    """Find an apparent horizon in volume data.

    The volume data must contain the inverse spatial metric, extrinsic
    curvature, and spatial Christoffel symbols. The data is assumed to be in the
    "inertial" frame.

    Arguments:
      h5_files: List of H5 files containing volume data or glob pattern.
      subfile_name: Name of the volume data subfile in the 'h5_files'.
      obs_id: Observation ID in the volume data.
      obs_time: Time of the observation.
      initial_guess: Initial guess for the horizon. Specify a
        'spectre.SphericalHarmonics.Strahlkorper'.
      fast_flow: Optional. FastFlow object that controls the horizon finder.
        If not specified, a FastFlow object with default parameters is used.
      output: Optional. H5 output file where the horizon Ylm coefficients will
        be written. Can be a new or existing file. Requires either
        'output_coeffs_subfile' or 'output_coords_subfile' is also specified,
        or both.
      output_coeffs_subfile: Optional. Name of the subfile in the H5 output file
        where the horizon Ylm coefficients will be written.
        These can be used to reconstruct the horizon, e.g. to initialize
        excisions in domains.
      output_coords_subfile: Optional. Name of the subfile in the H5 output file
        where the horizon coordinates will be written.
        These can be used for visualization.
      output_l_max: Optional. Maximum l-mode for the horizon Ylm coefficients
        written to the output file. Defaults to the l_max of the initial guess.
        Only used if 'output_coeffs_subfile' is specified.
      tensor_names: Optional. List of tensor names in the volume data that
        represent the inverse spatial metric, extrinsic curvature, and spatial
        Christoffel symbols, in this order. Defaults to ["InverseSpatialMetric",
        "ExtrinsicCurvature", "SpatialChristoffelSecondKind"].

    Returns: The Strahlkorper representing the horizon.
    """
    if not tensor_names:
        tensor_names = [
            "InverseSpatialMetric",
            "ExtrinsicCurvature",
            "SpatialChristoffelSecondKind",
        ]
    if fast_flow is None:
        fast_flow = FastFlow(
            FlowType.Fast,
            alpha=1.0,
            beta=0.5,
            abs_tol=1e-12,
            truncation_tol=0.01,
            divergence_tol=1.2,
            divergence_iter=5,
            max_its=100,
        )
    strahlkorper = initial_guess
    while True:
        l_mesh = fast_flow.current_l_mesh(strahlkorper)
        prolonged_strahlkorper = Strahlkorper(l_mesh, l_mesh, strahlkorper)
        (
            inv_spatial_metric,
            extrinsic_curvature,
            spatial_christoffel_second_kind,
        ) = interpolate_tensors_to_points(
            h5_files,
            subfile_name,
            observation_id=obs_id,
            target_points=cartesian_coords(prolonged_strahlkorper),
            tensor_names=tensor_names,
            tensor_types=[
                tnsr.II[DataVector, 3],
                tnsr.ii[DataVector, 3],
                tnsr.Ijj[DataVector, 3],
            ],
        )
        status, iter_info = fast_flow.iterate_horizon_finder(
            strahlkorper,
            upper_spatial_metric=inv_spatial_metric,
            extrinsic_curvature=extrinsic_curvature,
            christoffel_2nd_kind=spatial_christoffel_second_kind,
        )
        logger.debug(
            f"Horizon finder iteration {iter_info.iteration}: {status}."
            f" {iter_info.r_min:.3f} <= r <= {iter_info.r_max:.3f},"
            f" {iter_info.min_residual:.2e} <= residual <="
            f" {iter_info.max_residual:.2e}"
        )
        if status == Status.SuccessfulIteration:
            continue
        elif int(status) > 0:
            logger.info(
                f"Found horizon around {strahlkorper.expansion_center} with"
                f" {iter_info.r_min:.3f} <= r <= {iter_info.r_max:.3f}."
            )
            break
        else:
            raise RuntimeError(f"Horizon finder failed with status {status}.")
    # Write the horizon to a file and return it
    if output:
        assert output_coeffs_subfile or output_coords_subfile, (
            "Specify either 'output_coeffs_subfile' or 'output_coords_subfile'"
            " or both."
        )
        if output_coeffs_subfile:
            legend, ylm_data = ylm_legend_and_data(
                strahlkorper, obs_time, output_l_max or strahlkorper.l_max
            )
            with spectre_h5.H5File(output, "a") as output_file:
                datfile = output_file.try_insert_dat(
                    output_coeffs_subfile, legend, 0
                )
                datfile.append(ylm_data)
        if output_coords_subfile:
            vol_data = _strahlkorper_vol_data(strahlkorper)
            with spectre_h5.H5File(output, "a") as output_file:
                volfile = output_file.try_insert_vol(output_coords_subfile, 0)
                volfile.write_volume_data(obs_id, obs_time, vol_data)
    return strahlkorper


@click.command(name="find-horizon")
@open_volfiles_command(
    obs_id_required=True, multiple_vars=True, vars_required=False
)
@click.option(
    "--l-max",
    "-l",
    type=int,
    required=True,
    help="Max l-mode for the horizon search.",
)
@click.option(
    "--initial-radius",
    "-r",
    type=float,
    required=True,
    help="Initial coordinate radius of the horizon.",
)
@click.option(
    "--center",
    "-C",
    nargs=3,
    type=float,
    required=True,
    help="Coordinate center of the horizon.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    help=(
        "H5 output file where the horizon Ylm coefficients will be written. Can"
        " be a new or existing file."
    ),
)
@click.option(
    "--output-coeffs-subfile",
    help=(
        "Name of the subfile in the H5 output file where the horizon Ylm"
        " coefficients will be written. These can be used to reconstruct the"
        " horizon, e.g. to initialize excisions in domains."
    ),
)
@click.option(
    "--output-coords-subfile",
    help=(
        "Name of the subfile in the H5 output file where the horizon"
        " coordinates will be written. These can be used for visualization."
    ),
)
def find_horizon_command(l_max, initial_radius, center, vars, **kwargs):
    """Find an apparent horizon in volume data."""
    initial_guess = Strahlkorper(
        l_max=l_max, m_max=l_max, radius=initial_radius, center=center
    )
    find_horizon(
        initial_guess=initial_guess,
        tensor_names=vars,
        **kwargs,
    )


if __name__ == "__main__":
    find_horizon_command(help_option_names=["-h", "--help"])
