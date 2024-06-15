# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
from pathlib import Path
from typing import Optional, Union

import click
import yaml

from spectre.Pipelines.Bbh.FindHorizon import find_horizon, vec_to_string
from spectre.SphericalHarmonics import Strahlkorper
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.OpenVolfiles import open_volfiles
from spectre.Visualization.ReadH5 import select_observation
from spectre.Visualization.ReadInputFile import find_event

logger = logging.getLogger(__name__)


def postprocess_id(
    id_input_file_path: Union[str, Path],
    id_run_dir: Optional[Union[str, Path]] = None,
    horizon_l_max: int = 12,
    horizons_file: Optional[Union[str, Path]] = None,
    evolve: bool = False,
    pipeline_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Postprocess initial data after generation.

    This function is called automatically after the initial data has been
    generated (see the 'Next' section in the 'InitialData.yaml' input file), or
    manually by pointing the ID_INPUT_FILE_PATH to the input file of the initial
    data run. Also specify 'id_run_dir' if the initial data was run in a
    different directory than where the input file is.

    This function does the following:

    - Find apparent horizons in the data to determine quantities like the masses
      and spins of the black holes. These quantities are stored in the given
      'horizons_file' in subfiles 'Ah{A,B}.dat'. In addition, the horizon
      surface coordinates and coefficients are written to the 'horizons_file' in
      subfiles 'Ah{A,B}/Coordinates' and 'Ah{A,B}/Coefficients'.

    - Start the inspiral if 'evolve' is set to True.

    Arguments:
      id_input_file_path: Path to the input file of the initial data run.
      id_run_dir: Directory of the initial data run. Paths in the input file are
        relative to this directory. If not provided, the directory of the input
        file is used.
      horizon_l_max: Maximum l-mode for the horizon search.
      horizons_file: Path to the file where the horizon data is written to.
        Default is 'Horizons.h5' in the 'id_run_dir'.
      evolve: Evolve the initial data after postprocessing (default: False).
      pipeline_dir: Directory where steps in the pipeline are created.
        Required if 'evolve' is set to True.
    """
    # Read input file
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    x_B, x_A = id_input_file["Background"]["Binary"]["XCoords"]
    id_domain = id_input_file["DomainCreator"]["BinaryCompactObject"]
    excision_radius_A = id_domain[f"ObjectA"]["InnerRadius"]
    excision_radius_B = id_domain[f"ObjectB"]["InnerRadius"]
    volfile_name = id_input_file["Observers"]["VolumeFileName"]
    id_subfile_name = find_event("ObserveFields", id_input_file)["SubfileName"]

    # Find latest observation in output data
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    id_volfiles = glob.glob(str(Path(id_run_dir) / (volfile_name + "*.h5")))
    obs_id, _ = select_observation(
        open_volfiles(id_volfiles, id_subfile_name), step=-1
    )

    # Find horizons and write to the output file
    if not horizons_file:
        horizons_file = Path(id_run_dir) / "Horizons.h5"
    for object_label, xcoord, excision_radius in zip(
        ["AhA", "AhB"], [x_A, x_B], [excision_radius_A, excision_radius_B]
    ):
        _, horizon_quantities = find_horizon(
            id_volfiles,
            subfile_name=id_subfile_name,
            obs_id=obs_id,
            obs_time=0.0,
            initial_guess=Strahlkorper(
                l_max=horizon_l_max,
                radius=excision_radius * 1.5,
                center=[xcoord, 0.0, 0.0],
            ),
            output_surfaces_file=horizons_file,
            output_coeffs_subfile=f"{object_label}/Coefficients",
            output_coords_subfile=f"{object_label}/Coordinates",
            output_reductions_file=horizons_file,
            output_quantities_subfile=object_label,
        )
        logger.info(
            f"{object_label} has mass"
            f" {horizon_quantities['ChristodoulouMass']:g} and spin"
            f" {vec_to_string(horizon_quantities['DimensionlessSpinVector'])}."
        )
    logger.info(f"Horizons found and written to {horizons_file}.")

    # Start the inspiral from the ID if requested
    if evolve:
        from spectre.Pipelines.Bbh.Inspiral import start_inspiral

        start_inspiral(
            id_input_file_path,
            id_run_dir=id_run_dir,
            continue_with_ringdown=True,
            pipeline_dir=pipeline_dir,
            **scheduler_kwargs,
        )


@click.command(name="postprocess-id", help=postprocess_id.__doc__)
@click.argument(
    "id_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-i",
    "--id-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the initial data run. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the ID_INPUT_FILE_PATH",
)
@click.option(
    "--evolve",
    is_flag=True,
    help="Evolve the initial data after postprocessing.",
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(writable=True, path_type=Path),
    help="Directory where steps in the pipeline are created.",
)
@click.option(
    "--horizon-l-max",
    type=click.IntRange(0, None),
    help="Maximum l-mode for the horizon search.",
    default=12,
    show_default=True,
)
@click.option(
    "--horizons-file",
    type=click.Path(writable=True, path_type=Path),
    help="Path to the file where the horizon data is written to.",
    show_default="Horizons.h5 in the ID_RUN_DIR",
)
@scheduler_options
def postprocess_id_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    postprocess_id(**kwargs)


if __name__ == "__main__":
    postprocess_id_command(help_option_names=["-h", "--help"])
