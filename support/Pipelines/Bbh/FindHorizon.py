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
from spectre.PointwiseFunctions.GeneralRelativity.Surfaces import (
    horizon_quantities,
)
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


def _horizon_reduction_data(quantities):
    """Row of horizon quantities, can be written to an H5 dat file."""
    # Flatten vector quantities into a single row of values. So the spin is
    # returned as 3 columns with the x, y, and z components.
    legend = []
    reduction_data = []
    for key, value in quantities.items():
        if isinstance(value, float):
            legend.append(key)
            reduction_data.append(value)
        elif len(value) == 3:
            legend.extend([f"{key}_{xyz}" for xyz in "xyz"])
            reduction_data.extend(value)
        else:
            raise ValueError(
                f"Unsupported shape of horizon quantity {key}: {value}"
            )
    return legend, reduction_data


def vec_to_string(vec):
    """Convert a vector to a (readable) string."""
    return f"[{', '.join(f'{v:g}' for v in vec)}]"


def find_horizon(
    h5_files: Union[str, Sequence[str]],
    subfile_name: str,
    obs_id: int,
    obs_time: float,
    initial_guess: Strahlkorper,
    fast_flow: Optional[FastFlow] = None,
    output_surfaces_file: Optional[Union[str, Path]] = None,
    output_coeffs_subfile: Optional[str] = None,
    output_coords_subfile: Optional[str] = None,
    output_reductions_file: Optional[Union[str, Path]] = None,
    output_quantities_subfile: Optional[str] = None,
    output_l_max: Optional[int] = None,
    tensor_names: Optional[Sequence[str]] = None,
):
    """Find an apparent horizon in volume data.

    The volume data must contain the spatial metric, inverse spatial metric,
    extrinsic curvature, spatial Christoffel symbols, and spatial Ricci tensor.
    The data is assumed to be in the "inertial" frame.

    Arguments:
      h5_files: List of H5 files containing volume data or glob pattern.
      subfile_name: Name of the volume data subfile in the 'h5_files'.
      obs_id: Observation ID in the volume data.
      obs_time: Time of the observation.
      initial_guess: Initial guess for the horizon. Specify a
        'spectre.SphericalHarmonics.Strahlkorper'.
      fast_flow: Optional. FastFlow object that controls the horizon finder.
        If not specified, a FastFlow object with default parameters is used.
      output_surfaces_file: Optional. H5 output file where the horizon Ylm
        coefficients will be written. Can be a new or existing file. Requires
        either 'output_coeffs_subfile' or 'output_coords_subfile' is also
        specified, or both.
      output_coeffs_subfile: Optional. Name of the subfile in the
        'output_surfaces_file' where the horizon Ylm coefficients will be
        written. These can be used to reconstruct the horizon, e.g. to
        initialize excisions in domains.
      output_coords_subfile: Optional. Name of the subfile in the
        'output_surfaces_file' where the horizon coordinates will be written.
        These can be used for visualization.
      output_reductions_file: Optional. H5 output file where the reduction
        quantities on the horizon will be written, e.g. masses and spins.
        Can be a new or existing file. Requires 'output_quantities_subfile'
        is also specified.
      output_quantities_subfile: Optional. Name of the subfile in the
        'output_reductions_file' where the horizon quantities will be written,
        e.g. masses and spins.
      output_l_max: Optional. Maximum l-mode for the horizon Ylm coefficients
        written to the output file. Defaults to the l_max of the initial guess.
        Only used if 'output_coeffs_subfile' is specified.
      tensor_names: Optional. List of tensor names in the volume data that
        represent the spatial metric, inverse spatial metric, extrinsic
        curvature, spatial Christoffel symbols, and spatial Ricci tensor, in
        this order. Defaults to ["SpatialMetric", "InverseSpatialMetric",
        "ExtrinsicCurvature", "SpatialChristoffelSecondKind", "SpatialRicci"].

    Returns: The Strahlkorper representing the horizon, and a dictionary of
      horizon quantities (e.g. area, mass, spin, etc.).
    """
    # Validate input arguments
    if output_surfaces_file:
        assert output_coeffs_subfile or output_coords_subfile, (
            "Specify either 'output_coeffs_subfile' or 'output_coords_subfile'"
            " or both."
        )
    if not tensor_names:
        tensor_names = [
            "SpatialMetric",
            "InverseSpatialMetric",
            "ExtrinsicCurvature",
            "SpatialChristoffelSecondKind",
            "SpatialRicci",
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
            tensor_names=tensor_names[1:4],
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
            f" {iter_info.r_min:g} <= r <= {iter_info.r_max:g},"
            f" {iter_info.min_residual:.2e} <= residual <="
            f" {iter_info.max_residual:.2e}"
        )
        if status == Status.SuccessfulIteration:
            continue
        elif int(status) > 0:
            logger.info(
                "Found horizon around"
                f" {vec_to_string(strahlkorper.expansion_center)} with"
                f" {iter_info.r_min:g} <= r <= {iter_info.r_max:g}."
            )
            break
        else:
            raise RuntimeError(f"Horizon finder failed with status {status}.")
    # Compute horizon quantities
    # This is independent of the horizon find and could move into a separate
    # function, or disabled on request, if we ever need to find a horizon
    # without computing these quantities.
    (
        spatial_metric,
        inv_spatial_metric,
        extrinsic_curvature,
        spatial_christoffel_second_kind,
        spatial_ricci,
    ) = interpolate_tensors_to_points(
        h5_files,
        subfile_name,
        observation_id=obs_id,
        target_points=cartesian_coords(strahlkorper),
        tensor_names=tensor_names,
        tensor_types=[
            tnsr.ii[DataVector, 3],
            tnsr.II[DataVector, 3],
            tnsr.ii[DataVector, 3],
            tnsr.Ijj[DataVector, 3],
            tnsr.ii[DataVector, 3],
        ],
    )
    quantities = horizon_quantities(
        strahlkorper,
        spatial_metric=spatial_metric,
        inv_spatial_metric=inv_spatial_metric,
        extrinsic_curvature=extrinsic_curvature,
        spatial_christoffel_second_kind=spatial_christoffel_second_kind,
        spatial_ricci=spatial_ricci,
    )
    # Write the horizon to a file and return it
    if output_surfaces_file:
        if Path(output_surfaces_file).suffix not in [".h5", ".hdf5"]:
            output_surfaces_file += ".h5"
        if output_coeffs_subfile:
            legend, ylm_data = ylm_legend_and_data(
                strahlkorper, obs_time, output_l_max or strahlkorper.l_max
            )
            with spectre_h5.H5File(
                str(output_surfaces_file), "a"
            ) as output_file:
                datfile = output_file.try_insert_dat(
                    output_coeffs_subfile, legend, 0
                )
                datfile.append(ylm_data)
        if output_coords_subfile:
            vol_data = _strahlkorper_vol_data(strahlkorper)
            with spectre_h5.H5File(
                str(output_surfaces_file), "a"
            ) as output_file:
                volfile = output_file.try_insert_vol(output_coords_subfile, 0)
                volfile.write_volume_data(obs_id, obs_time, vol_data)
    if output_reductions_file:
        assert output_quantities_subfile, (
            "Specify 'output_quantities_subfile' if 'output_reductions_file'"
            " is specified."
        )
        if Path(output_reductions_file).suffix not in [".h5", ".hdf5"]:
            output_reductions_file += ".h5"
        legend, reduction_data = _horizon_reduction_data(quantities)
        with spectre_h5.H5File(str(output_reductions_file), "a") as output_file:
            datfile = output_file.try_insert_dat(
                output_quantities_subfile, legend, 0
            )
            datfile.append(reduction_data)
    return strahlkorper, quantities


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
    "--output-surfaces-file",
    type=click.Path(writable=True),
    help=(
        "H5 output file where the horizon Ylm coefficients will be written. Can"
        " be a new or existing file."
    ),
)
@click.option(
    "--output-coeffs-subfile",
    help=(
        "Name of the subfile in the 'output_surfaces_file' where the horizon"
        " Ylm coefficients will be written. These can be used to reconstruct"
        " the horizon, e.g. to initialize excisions in domains."
    ),
)
@click.option(
    "--output-coords-subfile",
    help=(
        "Name of the subfile in the 'output_surfaces_file' where the horizon"
        " coordinates will be written. These can be used for visualization."
    ),
)
@click.option(
    "--output-reductions-file",
    type=click.Path(writable=True),
    help=(
        "H5 output file where the reduction quantities on the horizon will be"
        " written, e.g. masses and spins. Can be a new or existing file."
    ),
)
@click.option(
    "--output-quantities-subfile",
    help=(
        "Name of the subfile in the 'output_reductions_file' where the horizon"
        " quantities will be written, e.g. masses and spins."
    ),
)
def find_horizon_command(l_max, initial_radius, center, vars, **kwargs):
    """Find an apparent horizon in volume data."""
    initial_guess = Strahlkorper(
        l_max=l_max, radius=initial_radius, center=center
    )
    horizon, quantities = find_horizon(
        initial_guess=initial_guess,
        tensor_names=vars,
        **kwargs,
    )

    # Output horizon quantities
    import rich.table

    table = rich.table.Table(show_header=False, box=None)
    for name, value in quantities.items():
        table.add_row(
            name,
            (
                f"{value:g}"
                if isinstance(value, float)
                else vec_to_string(value)
            ),
        )
    rich.print(table)


if __name__ == "__main__":
    find_horizon_command(help_option_names=["-h", "--help"])
