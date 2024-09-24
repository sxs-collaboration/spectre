# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np

import spectre.IO.H5 as spectre_h5
import spectre.SurfaceFinder
from spectre.Domain import (
    ElementId,
    block_logical_coordinates,
    deserialize_domain,
    deserialize_functions_of_time,
    element_logical_coordinates,
)
from spectre.IO.H5.IterElements import iter_elements
from spectre.Pipelines.Bbh.FindHorizon import _strahlkorper_vol_data
from spectre.SphericalHarmonics import (
    Strahlkorper,
    cartesian_coords,
    ylm_legend_and_data,
)
from spectre.Visualization.OpenVolfiles import (
    open_volfiles,
    open_volfiles_command,
)

logger = logging.getLogger(__name__)


def find_radial_surface(
    h5_files: Sequence[str],
    subfile_name: str,
    obs_id: int,
    obs_time: float,
    var_name: str,
    target: float,
    initial_guess: Strahlkorper,
    output_surfaces_file: Optional[Union[str, Path]] = None,
    output_coeffs_subfile: Optional[str] = None,
    output_coords_subfile: Optional[str] = None,
):
    """Find a radial surface where a variable equals a target value.

    The surface is found by searching along radial rays that are defined by the
    angular collocation points of the 'initial_guess' Strahlkorper. Only blocks
    that intersect the initial guess and their radial neighbors are considered,
    and only those neighbors that have a distorted frame. These blocks are
    assumed to be wedges, so their third logical coordinate is radial.

    This function is useful, for example, for finding the surface of a star
    and then deforming the domain to match the surface.

    \f
    Arguments:
      h5_files: The H5 files containing the volume data.
      subfile_name: Name of the volume data subfile in the 'h5_files'.
      obs_id: Observation ID in the volume data.
      obs_time: Time of the observation.
      var_name: Name of the variable in the volume data to search for the
        surface.
      target: Value that the variable takes that defines the surface.
      initial_guess: Initial guess for the surface. Only blocks that intersect
        this Strahlkorper and their radial neighbors are considered. These
        blocks must be wedges.
      output_surfaces_file: Optional. H5 output file where the surface Ylm
        coefficients will be written. Can be a new or existing file. Requires
        either 'output_coeffs_subfile' or 'output_coords_subfile' is also
        specified, or both.
      output_coeffs_subfile: Optional. Name of the subfile in the
        'output_surfaces_file' where the surface Ylm coefficients will be
        written. These can be used to reconstruct the surface, e.g. to
        deform the domain.
      output_coords_subfile: Optional. Name of the subfile in the
        'output_surfaces_file' where the surface coordinates will be written.
        These can be used for visualization.
    """
    # Validate input arguments
    if output_surfaces_file:
        assert output_coeffs_subfile or output_coords_subfile, (
            "Specify either 'output_coeffs_subfile' or 'output_coords_subfile'"
            " or both."
        )
    # Deserialize domain and functions of time
    for volfile in open_volfiles(h5_files, subfile_name, obs_id):
        dim = volfile.get_dimension()
        domain = deserialize_domain[dim](volfile.get_domain(obs_id))
        time = volfile.get_observation_value(obs_id)
        functions_of_time = (
            deserialize_functions_of_time(volfile.get_functions_of_time(obs_id))
            if domain.is_time_dependent()
            else None
        )
        break
    # Map the initial guess through the domain. We need the angular logical
    # coordinates to define the rays along which we search for the surface.
    initial_guess_coords = cartesian_coords(initial_guess)
    block_logical_coords = block_logical_coordinates(
        domain, initial_guess_coords, time, functions_of_time
    )
    num_rays = len(block_logical_coords)
    # Find the surface
    surface_radii = np.empty(num_rays)
    surface_radii.fill(np.nan)
    filled = np.zeros(num_rays, dtype=bool)
    for element, data in iter_elements(
        open_volfiles(h5_files, subfile_name, obs_id),
        obs_ids=obs_id,
        tensor_components=[var_name],
    ):
        # Find the radial rays that go through this element
        angular_coords = []
        offsets = []
        for i, block_logical_coord in enumerate(block_logical_coords):
            if filled[i]:
                continue
            block_id = block_logical_coord.id.get_index()
            # Radial neighbors that are wedges are also valid because they keep
            # the angular logical coordinates the same
            block_neighbors = [
                block_neighbor_id
                for direction, block_neighbor_id in domain.blocks[
                    block_id
                ].neighbors.items()
                if direction.dimension == 2
                and domain.blocks[block_neighbor_id].has_distorted_frame()
            ]
            valid_blocks = [block_id, *block_neighbors]
            if element.id.block_id not in valid_blocks:
                continue
            angular_block_logical = np.array(block_logical_coord.data)[:2]
            angular_element_id = ElementId[2](
                element.id.block_id, element.id.segment_ids[:2]
            )
            angular_element_logical = element_logical_coordinates(
                angular_block_logical, angular_element_id
            )
            if angular_element_logical is None:
                continue
            angular_coords.append(angular_element_logical)
            offsets.append(i)
        if len(angular_coords) == 0:
            continue
        # Find the surface in this element along each ray
        angular_coords = np.array(angular_coords).T
        surface_logical_radii = spectre.SurfaceFinder.find_radial_surface(
            data[0], target, element.mesh, angular_coords
        )
        # For those rays where we found the surface in this element, map the
        # surface radius to inertial coordinates
        for offset, surface_logical_radius, angular_element_logical in zip(
            offsets, surface_logical_radii, angular_coords.T
        ):
            if surface_logical_radius is None:
                continue
            surface_logical_point = [
                *angular_element_logical,
                surface_logical_radius,
            ]
            surface_radii[offset] = np.linalg.norm(
                np.array(
                    element.map(
                        np.array(surface_logical_point).reshape((3, 1)),
                        time,
                        functions_of_time,
                    )
                )[:, 0]
            )
            filled[offset] = True
        if np.all(filled):
            break
    if not np.all(filled):
        missing_points = np.array(initial_guess_coords).T[~filled]
        raise ValueError(
            f"Unable to find the surface where {var_name} = {target:g} with"
            f" center {initial_guess.expansion_center}. Missing values for"
            f" {len(missing_points)} / {len(filled)} radial rays. One of the"
            f" missing rays goes through: {missing_points[0]}"
        )
    surface = Strahlkorper(
        l_max=initial_guess.l_max,
        m_max=initial_guess.m_max,
        radius_at_collocation_points=surface_radii,
        center=initial_guess.expansion_center,
    )
    logger.info(
        f"Found radial surface where {var_name} = {target:g} with average"
        f" radius {surface.average_radius:g} and center"
        f" {surface.expansion_center}."
    )
    # Write the surface to a file and return it
    if output_surfaces_file:
        if Path(output_surfaces_file).suffix not in [".h5", ".hdf5"]:
            output_surfaces_file += ".h5"
        if output_coeffs_subfile:
            legend, ylm_data = ylm_legend_and_data(
                surface, obs_time, surface.l_max
            )
            with spectre_h5.H5File(
                str(output_surfaces_file), "a"
            ) as output_file:
                datfile = output_file.try_insert_dat(
                    output_coeffs_subfile, legend, 0
                )
                datfile.append(ylm_data)
        if output_coords_subfile:
            vol_data = _strahlkorper_vol_data(surface)
            with spectre_h5.H5File(
                str(output_surfaces_file), "a"
            ) as output_file:
                volfile = output_file.try_insert_vol(output_coords_subfile, 0)
                volfile.write_volume_data(obs_id, obs_time, vol_data)
    return surface


@click.command(name="find-radial-surface", help=find_radial_surface.__doc__)
@open_volfiles_command(
    obs_id_required=True, multiple_vars=False, vars_required=True
)
@click.option(
    "--target",
    "-t",
    type=float,
    required=True,
    help="Target value for the surface.",
)
@click.option(
    "--l-max",
    "-l",
    type=int,
    required=True,
    help="Max l-mode for the Ylm representation of the surface.",
)
@click.option(
    "--initial-radius",
    "-r",
    type=float,
    required=True,
    help="Coordinate radius of the spherical initial guess for the surface.",
)
@click.option(
    "--center",
    "-C",
    nargs=3,
    type=float,
    required=True,
    help="Coordinate center of the Ylm representation of the surface.",
)
@click.option(
    "--output-surfaces-file",
    type=click.Path(writable=True),
    help=(
        "H5 output file where the surface Ylm coefficients will be written. Can"
        " be a new or existing file."
    ),
)
@click.option(
    "--output-coeffs-subfile",
    help=(
        "Name of the subfile in the 'output_surfaces_file' where the surface"
        " Ylm coefficients will be written. These can be used to reconstruct"
        " the surface, e.g. to deform the domain."
    ),
)
@click.option(
    "--output-coords-subfile",
    help=(
        "Name of the subfile in the 'output_surfaces_file' where the surface"
        " coordinates will be written. These can be used for visualization."
    ),
)
def find_radial_surface_command(l_max, initial_radius, center, **kwargs):
    initial_guess = Strahlkorper(
        l_max=l_max, radius=initial_radius, center=center
    )
    find_radial_surface(initial_guess=initial_guess, **kwargs)


if __name__ == "__main__":
    find_radial_surface_command(help_option_names=["-h", "--help"])
