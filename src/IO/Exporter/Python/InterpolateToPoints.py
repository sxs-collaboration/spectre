# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import click
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.IO.Exporter import interpolate_to_points
from spectre.Visualization.OpenVolfiles import (
    open_volfiles_command,
    parse_points,
)

logger = logging.getLogger(__name__)


@click.command(name="interpolate-to-points")
@open_volfiles_command(obs_id_required=True, multiple_vars=True)
@click.option(
    "--target-coords-file",
    "-t",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help=(
        "Text file with target coordinates to interpolate to. "
        "Must have 'dim' columns with Cartesian coordinates. "
        "Rows enumerate points. "
        "Can be the output of 'numpy.savetxt'."
    ),
)
@click.option(
    "--target-coords",
    "-p",
    multiple=True,
    callback=parse_points,
    help=(
        "List target coordinates explicitly, e.g. '0,0,0'. "
        "Can be specified multiple times to quickly interpolate "
        "to a couple of target points."
    ),
)
@click.option(
    "--extrapolate-into-excisions",
    is_flag=True,
    help=(
        "Enables extrapolation into excision regions of the domain. "
        "This can be useful to fill the excision region with "
        "(constraint-violating but smooth) data so it can be imported into "
        "moving puncture codes."
    ),
)
@click.option(
    "--output", "-o", type=click.Path(writable=True), help="Output text file"
)
@click.option(
    "--delimiter",
    default=None,
    show_default="whitespace",
    help=(
        "Delimiter separating columns for both the "
        "'--target-coords-file' and the '--output' file."
    ),
)
@click.option(
    "--num-threads",
    "-j",
    type=int,
    show_default="all available cores",
    help=(
        "Number of threads to use for interpolation. Only available if compiled"
        " with OpenMP. Parallelization is over volume data files, so this only"
        " has an effect if multiple files are specified."
    ),
)
def interpolate_to_points_command(
    h5_files,
    subfile_name,
    obs_id,
    obs_time,
    vars,
    target_coords,
    target_coords_file,
    output,
    delimiter,
    **kwargs,
):
    """Interpolate volume data to target coordinates."""
    # Load target coords from file
    if (target_coords is None) == (target_coords_file is None):
        raise click.UsageError(
            "Specify either '--target-coords' / '-p' or "
            "'--target-coords-file' / '-t'."
        )
    if target_coords_file:
        target_coords = np.loadtxt(
            target_coords_file, ndmin=2, delimiter=delimiter
        )
    dim = target_coords.shape[1]

    # Interpolate!
    interpolated_data = np.array(
        interpolate_to_points(
            h5_files,
            subfile_name=subfile_name,
            observation_id=obs_id,
            tensor_components=vars,
            target_points=target_coords.T,
            **kwargs,
        )
    )

    # Output result
    column_names = ["X", "Y", "Z"][:dim] + list(vars)
    column_data = np.hstack([target_coords, interpolated_data.T])
    if output:
        np.savetxt(
            output,
            column_data,
            delimiter=delimiter or " ",
            header=(
                f"t = {obs_time:g}\n" + (delimiter or " ").join(column_names)
            ),
        )
    else:
        import rich.table

        table = rich.table.Table(*column_names, box=None)
        for row in column_data:
            table.add_row(*list(map(str, row)))
        rich.print(table)
        rich.print(f"(t = {obs_time:g})")


if __name__ == "__main__":
    interpolate_to_points_command(help_option_names=["-h", "--help"])
