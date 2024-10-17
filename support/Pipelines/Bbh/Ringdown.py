# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.IO.H5 as spectre_h5
from spectre.Evolution.Ringdown.ComputeAhCCoefsInRingdownDistortedFrame import (
    compute_ahc_coefs_in_ringdown_distorted_frame,
)

# next import out of order to avoid Unrecognized PUP::able::PUP_ID error
from spectre.Evolution.Ringdown.FunctionsOfTimeFromVolume import (
    functions_of_time_from_volume,
)
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

RINGDOWN_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Ringdown.yaml"


def ringdown_parameters(
    inspiral_input_file: dict,
    inspiral_run_dir: Union[str, Path],
    fot_vol_subfile: str,
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine ringdown parameters from the inspiral.

    These parameters fill the 'RINGDOWN_INPUT_FILE_TEMPLATE'.

    Arguments:
      inspiral_input_file: Inspiral input file as a dictionary.
      id_run_dir: Directory of the inspiral run. Paths in the input file
        are relative to this directory.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """
    return {
        # Initial data files
        "IdFileGlob": str(
            Path(inspiral_run_dir).resolve()
            / (inspiral_input_file["Observers"]["VolumeFileName"] + "*.h5")
        ),
        "IdFileGlobSubgroup": fot_vol_subfile,
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def start_ringdown(
    inspiral_run_dir: Union[str, Path],
    number_of_steps: int,
    match_time: float,
    settling_timescale: float,
    zero_coefs: float,
    refinement_level: int,
    polynomial_order: int,
    inspiral_input_file: Optional[Union[str, Path]] = None,
    ahc_reductions_path: Optional[Union[str, Path]] = None,
    ahc_subfile: str = "ObservationAhC_Ylm",
    fot_vol_h5_path: Optional[Union[str, Path]] = None,
    fot_vol_subfile: str = "ForContinuation",
    path_to_output_h5: Optional[Union[str, Path]] = None,
    output_subfile_prefix: str = "Distorted",
    ringdown_input_file_template: Union[
        str, Path
    ] = RINGDOWN_INPUT_FILE_TEMPLATE,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Schedule a ringdown simulation from the inspiral.

    Point the inspiral_run_dir to the last inspiral segment. Also specify
    'inspiral_input_file' if the simulation was run in a different directory
    than where the input file is. Parameters for the ringdown will be determined
    from the inspiral and inserted into the
    'ringdown_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    Here 'parameters for the ringdown' includes the information needed to
    initialize the time-dependent maps, including the shape map. Common horizon
    shape coefficients in the ringdown distorted frame will be written to
    disk that the ringdown input file will point to.

    Arguments:
        inspiral_run_dir: Path to the last segment in the inspiral run
        directory.
        number_of_steps: The number of steps from the last time in the
        simulation to look for AhC finds.
        match_time: The time to match the time dependent maps at.
        settling_timescale: The settling timescale for the rotation and
        expansion maps.
        zero_coefs: Coefficients to set to 0.0
        refinement_level: The initial H refinement level for ringdown.
        polynomial_order: The initial P refinement level for ringdown.
        inspiral_input_file: The input file used for during the Inspiral,
        defaults to the Inspiral.yaml inside the inspiral_run_dir.
        ahc_reductions_path: The full path to the BbhReductions file that
        contains AhC data, defaults to BbhReductions.h5 in the inspiral_run_dir.
        ahc_subfile: Subfile containing reduction data at times of AhC finds,
        defaults to 'ObservationAhC_Ylm'.
        fot_vol_h5_path: The full path to any volume data containing the
        functions of time at the time of AhC finds, defaults to BbhVolume0.h5 in
        the inspiral_run_dir.
        fot_vol_subfile: Subfile containing volume data where at times of AhC
        finds, defaults to 'ForContinuation'.
        path_to_output_h5: H5 file to output horizon coefficients needed for
        Ringdown.
        output_subfile_prefix: Subfile prefix for output data, defaults to
        'Distorted'.
        ringdown_input_file_template: Yaml to insert ringdown coefficients into.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files. In particular, the ringdown BBH pipline has"
        " been tested for a q=1, spin=0 quasicircular inspiral but does not"
        " yet support accounting for a nonzero translation map in the inspiral"
        " (necessary for unequal-mass mergers.)"
    )
    # Determine ringdown parameters from inspiral
    # Resolve and set correct files/paths.
    if inspiral_input_file is None:
        inspiral_input_file = inspiral_run_dir / "Inspiral.yaml"

    if ahc_reductions_path is None:
        ahc_reductions_path = inspiral_run_dir / "BbhReductions.h5"

    if fot_vol_h5_path is None:
        fot_vol_h5_path = inspiral_run_dir / "BbhVolume0.h5"

    with open(inspiral_input_file, "r") as open_input_file:
        _, inspiral_input_file = yaml.safe_load_all(open_input_file)

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if pipeline_dir and not segments_dir and not run_dir:
        segments_dir = pipeline_dir / "003_Ringdown"

    if path_to_output_h5 is None:
        path_to_output_h5 = pipeline_dir / "RingdownDistortedCoefs.h5"

    ringdown_params = ringdown_parameters(
        inspiral_input_file,
        inspiral_run_dir,
        fot_vol_subfile,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )

    # Compute ringdown shape coefficients and function of time info
    # for ringdown
    with spectre_h5.H5File(str(fot_vol_h5_path), "r") as h5file:
        if fot_vol_subfile.split(".")[-1] == "vol":
            fot_vol_subfile = fot_vol_subfile.split(".")[0]
        volfile = h5file.get_vol("/" + fot_vol_subfile)
        obs_ids = volfile.list_observation_ids()
        fot_times = np.array(list(map(volfile.get_observation_value, obs_ids)))
        which_obs_id = np.argmin(np.abs(fot_times - match_time))

        logger.info("Desired match time: " + str(match_time))
        logger.info("Selected ObservationID: " + str(which_obs_id))
        logger.info("Selected match time: " + str(fot_times[which_obs_id]))

    match_time = fot_times[which_obs_id]

    fot_vol_expansion, fot_vol_expansion_outer_boundary, fot_vol_rotation = (
        functions_of_time_from_volume(
            str(fot_vol_h5_path), fot_vol_subfile, match_time, which_obs_id
        )
    )

    coefs, fot_info = compute_ahc_coefs_in_ringdown_distorted_frame(
        str(ahc_reductions_path),
        ahc_subfile,
        fot_vol_expansion,
        fot_vol_expansion_outer_boundary,
        fot_vol_rotation,
        str(path_to_output_h5),
        output_subfile_prefix,
        number_of_steps,
        match_time,
        settling_timescale,
        zero_coefs,
    )
    logger.info("Obtained ringdown coefs")

    ringdown_params["MatchTime"] = match_time
    ringdown_params["ShapeMapLMax"] = fot_info["Lmax"]
    ringdown_params["PathToAhCCoefsH5File"] = path_to_output_h5
    ringdown_params["AhCCoefsSubfilePrefix"] = output_subfile_prefix
    ringdown_params["Rotation0"] = fot_info["Rotation0"][0]
    ringdown_params["Rotation1"] = fot_info["Rotation1"][0]
    ringdown_params["Rotation2"] = fot_info["Rotation2"][0]
    ringdown_params["Rotation3"] = fot_info["Rotation3"][0]
    ringdown_params["dtRotation0"] = fot_info["Rotation0"][1]
    ringdown_params["dtRotation1"] = fot_info["Rotation1"][1]
    ringdown_params["dtRotation2"] = fot_info["Rotation2"][1]
    ringdown_params["dtRotation3"] = fot_info["Rotation3"][1]
    ringdown_params["dt2Rotation0"] = fot_info["Rotation0"][2]
    ringdown_params["dt2Rotation1"] = fot_info["Rotation1"][2]
    ringdown_params["dt2Rotation2"] = fot_info["Rotation2"][2]
    ringdown_params["dt2Rotation3"] = fot_info["Rotation3"][2]
    ringdown_params["ExpansionOuterBdry"] = fot_info["ExpansionOuterBdry"][0]
    ringdown_params["dtExpansionOuterBdry"] = fot_info["ExpansionOuterBdry"][1]
    ringdown_params["dt2ExpansionOuterBdry"] = fot_info["ExpansionOuterBdry"][2]
    # To avoid interpolation errors, put outer boundary of ringdown domain
    # slightly inside the outer boundary of the inspiral domain
    ringdown_params["OuterBdryRadius"] = (
        inspiral_input_file["DomainCreator"]["BinaryCompactObject"][
            "OuterShell"
        ]["Radius"]
        - 1.0e-4
    )
    # Give the black hole 200M to relax, and then the light travel time
    # to the outer boundary for the gravitational waves to leave the domain
    ringdown_params["FinalTime"] = (
        match_time + ringdown_params["OuterBdryRadius"] + 200.0
    )

    logger.info(f"Ringdown parameters: {pretty_repr(ringdown_params)}")

    # Add additional parameters to substitute in ringdown template
    # Primarily, these will initialize functions of time
    extra_params = {}

    # Schedule!
    return schedule(
        ringdown_input_file_template,
        **ringdown_params,
        **scheduler_kwargs,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="start-ringdown", help=start_ringdown.__doc__)
@click.argument(
    "inspiral_run_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-i",
    "--inspiral-input-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    help="Path to Inspiral yaml, defaults to Inspiral.yaml in directory given.",
)
@click.option(
    "--ahc-reductions-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    help=(
        "Path to reduction file containing AhC coefs, defualts to"
        " 'BbhReductions.h5' in directory given."
    ),
)
@click.option(
    "--ahc-subfile",
    type=str,
    default="ObservationAhC_Ylm",
    help=(
        "Subfile path name in reduction data containing AhC coefs, defaults to"
        " 'ObservationAhC_Ylm'"
    ),
)
@click.option(
    "--fot-vol-h5-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    help=(
        "Path to volume data file containing funcs of time, defaults to"
        " 'BbhVolume0.h5' in directory given."
    ),
)
@click.option(
    "--fot-vol-subfile",
    type=str,
    default="ForContinuation",
    help=(
        "Subfile in volume data with funcs of time at different times,"
        " defaults to 'ForContinuation'."
    ),
)
@click.option(
    "--path-to-output-h5",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    help="Output h5 file for shape coefs",
)
@click.option(
    "--output-subfile-prefix",
    type=str,
    default="Distorted",
    help="Output subfile prefix for AhC coefs",
)
@click.option(
    "--number-of-steps",
    type=int,
    required=True,
    help="Number of steps from end to look for AhC data",
)
@click.option(
    "--match-time",
    required=True,
    type=float,
    help="Desired match time (volume data must contain data at this time)",
)
@click.option(
    "--settling-timescale",
    required=True,
    type=float,
    help="Damping timescale for settle to const",
)
@click.option(
    "--zero-coefs",
    type=float,
    default=None,
    help="What value of coefs to zero below. None means don't zero any",
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=2,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=11,
    show_default=True,
)
@click.option(
    "--ringdown-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=RINGDOWN_INPUT_FILE_TEMPLATE,
    help="Input file template for the ringdown.",
    show_default=True,
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory where steps in the pipeline are created.",
)
@scheduler_options
def start_ringdown_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    start_ringdown(**kwargs)


if __name__ == "__main__":
    start_ringdown_command(help_option_names=["-h", "--help"])
