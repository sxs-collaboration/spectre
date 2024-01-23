# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import click
import numpy as np
import rich

import spectre.IO.H5 as spectre_h5
from spectre.IO.Exporter import interpolate_to_points

logger = logging.getLogger(__name__)


def parse_points(ctx, param, values):
    if not values:
        return None
    points = [list(map(float, value.split(","))) for value in values]
    dim = len(points[0])
    if any([len(point) != dim for point in points]):
        raise click.BadParameter(
            "All specified points must have the same dimension"
        )
    return np.array(points)


@click.command(name="interpolate-to-points")
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name",
    "-d",
    help=(
        "Name of subfile within h5 file containing volume data to interpolate."
    ),
)
@click.option(
    "--list-vars",
    "-l",
    is_flag=True,
    help="Print available variables and exit.",
)
@click.option(
    "--var",
    "-y",
    "vars",
    multiple=True,
    help=(
        "Variables to interpolate. List any tensor components "
        "in the volume data file, such as 'Shift_x'."
    ),
)
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
# Note: support for files with different observations can be added if needed by
# porting `Visualization.ReadH5:select_observation` to C++. That function would
# also support a `time` option, as an alternative to `step`.
@click.option(
    "--step",
    type=int,
    help=(
        "Observation step number. Specify '-1' for the last step in the file. "
        "All files must contain the same set of observations. Support for "
        "files with different observations (e.g. from multiple segments of a "
        "simulation) can be added if needed."
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
    list_vars,
    vars,
    target_coords,
    target_coords_file,
    output,
    step,
    delimiter,
    num_threads,
):
    """Interpolate volume data to target coordinates."""
    # Script should be a noop if input files are empty
    if not h5_files:
        return

    # Open first H5 file to get some info
    open_h5_file = spectre_h5.H5File(h5_files[0], "r")

    # Print available subfile names and exit
    if not subfile_name:
        import rich.columns

        available_subfiles = open_h5_file.all_vol_files()
        if len(available_subfiles) == 1:
            subfile_name = available_subfiles[0]
        else:
            rich.print(rich.columns.Columns(available_subfiles))
            return

    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]
    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name

    volfile = open_h5_file.get_vol(subfile_name)
    obs_ids = volfile.list_observation_ids()
    obs_values = list(map(volfile.get_observation_value, obs_ids))
    dim = volfile.get_dimension()

    # Select observation
    if step is None:
        raise click.UsageError(
            f"Must specify '--step' (in [0, {len(obs_ids) - 1}], or -1)."
        )
    obs_id = obs_ids[step]

    # Print available variables and exit
    all_vars = volfile.list_tensor_components(obs_id)
    if list_vars or not vars:
        import rich.columns

        rich.print(rich.columns.Columns(all_vars))
        return
    for var in vars:
        if var not in all_vars:
            raise click.UsageError(
                f"Unknown variable '{var}'. Available variables are: {all_vars}"
            )

    # Close the H5 file because we're done with preprocessing
    open_h5_file.close()

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
    if target_coords.shape[1] != dim:
        raise click.UsageError(
            f"Target coordinates must have dimension {dim} consistent with"
            f" volume data files, but have dimension {target_coords.shape[1]}."
        )

    # Interpolate!
    interpolated_data = np.array(
        interpolate_to_points(
            h5_files,
            subfile_name=subfile_name,
            observation_step=step,
            tensor_components=vars,
            target_points=target_coords.T,
            num_threads=num_threads,
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
                f"t = {obs_values[step]:g}\n"
                + (delimiter or " ").join(column_names)
            ),
        )
    else:
        import rich.table

        table = rich.table.Table(*column_names, box=None)
        for row in column_data:
            table.add_row(*list(map(str, row)))
        rich.print(table)
        rich.print(f"(t = {obs_values[step]:g})")


if __name__ == "__main__":
    interpolate_to_points_command(help_option_names=["-h", "--help"])
