# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import click
import yaml
from rich.pretty import pretty_repr

import spectre.IO.H5 as spectre_h5
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

INSPIRAL_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Inspiral.yaml"


# These parameters come from empirically tested values in SpEC and SpECTRE
def _control_system_params(
    mass_left: float,
    mass_right: float,
    spin_magnitude_left: float,
    spin_magnitude_right: float,
) -> dict:
    total_mass = mass_left + mass_right
    mass_ratio = max(mass_left, mass_right) / min(mass_left, mass_right)
    if spin_magnitude_left > 0.9 or spin_magnitude_right > 0.9:
        damping_time_base = 0.1
        decrease_threshold_base = 2e-4
        max_damping_timescale = 10.0
    else:
        damping_time_base = 0.2
        decrease_threshold_base = 2e-3
        max_damping_timescale = 20.0

    kinematic_timescale = damping_time_base * total_mass
    decrease_threshold = (
        0.1 * decrease_threshold_base / (mass_ratio + 1.0 / mass_ratio)
    )
    increase_threshold_fraction = 0.25

    return {
        "MaxDampingTimescale": max_damping_timescale,
        "KinematicTimescale": kinematic_timescale,
        "SizeATimescale": damping_time_base * 0.2 * mass_right,
        "SizeBTimescale": damping_time_base * 0.2 * mass_left,
        "ShapeATimescale": 5.0 * kinematic_timescale,
        "ShapeBTimescale": 5.0 * kinematic_timescale,
        "SizeIncreaseThreshold": 1e-3,
        "DecreaseThreshold": decrease_threshold,
        "IncreaseThreshold": increase_threshold_fraction * decrease_threshold,
        "SizeBMaxTimescale": 10 if spin_magnitude_left > 0.9 else 20,
        "SizeAMaxTimescale": 10 if spin_magnitude_right > 0.9 else 20,
    }


def _constraint_damping_params(
    mass_left: float,
    mass_right: float,
    initial_separation: float,
) -> dict:
    total_mass = mass_left + mass_right
    return {
        "Gamma0Constant": 0.001 / total_mass,
        "Gamma0LeftAmplitude": 4.0 / mass_left,
        "Gamma0LeftWidth": 7.0 * mass_left,
        "Gamma0RightAmplitude": 4.0 / mass_right,
        "Gamma0RightWidth": 7.0 * mass_right,
        "Gamma0OriginAmplitude": 0.075 / total_mass,
        "Gamma0OriginWidth": 2.5 * initial_separation,
        "Gamma1Width": 10.0 * initial_separation,
    }


def inspiral_parameters(
    id_input_file: dict,
    id_run_dir: Union[str, Path],
    id_horizons_path: Optional[Union[str, Path]],
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine inspiral parameters from SpECTRE initial data.

    These parameters fill the 'INSPIRAL_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_input_file: Initial data input file as a dictionary.
      id_run_dir: Directory of the initial data run. Paths in the input file
        are relative to this directory.
      id_horizons_path: Path to H5 file containing information about the
        horizons in the ID (e.g. mass, spin, spherical harmonic coefficients).
        If this is 'None', the default is the 'Horizons.h5' file inside
        'id_run_dir'.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """
    id_domain_creator = id_input_file["DomainCreator"]["BinaryCompactObject"]
    id_binary = id_input_file["Background"]["Binary"]

    # ID parameters
    horizons_filename = (
        Path(id_horizons_path)
        if id_horizons_path is not None
        else Path(id_run_dir) / "Horizons.h5"
    )
    if not horizons_filename.is_file():
        raise ValueError(
            f"The ID horizons path ({str(horizons_filename.resolve())}) does"
            " not exist. If there is no 'Horizons.h5' file in your ID"
            " directory, run 'spectre bbh postprocess-id' on the ID to"
            " generate it."
        )
    initial_separation = (
        id_domain_creator["ObjectA"]["XCoord"]
        - id_domain_creator["ObjectB"]["XCoord"]
    )
    with spectre_h5.H5File(
        str(horizons_filename.resolve()), "r"
    ) as horizons_file:
        aha_quantities = to_dataframe(horizons_file.get_dat("AhA.dat")).iloc[-1]
        mass_right = aha_quantities["ChristodoulouMass"]
        spin_magnitude_right = aha_quantities["DimensionlessSpinMagnitude"]

        horizons_file.close_current_object()
        ahb_quantities = to_dataframe(horizons_file.get_dat("AhB.dat")).iloc[-1]
        mass_left = ahb_quantities["ChristodoulouMass"]
        spin_magnitude_left = ahb_quantities["DimensionlessSpinMagnitude"]

    # Uncomment when we are confident that we can make the total mass 1. For
    # now, allow total mass != 1
    # total_mass = (
    #     horizon_quantities_A["ChristodoulouMass"]
    #     + horizon_quantities_B["ChristodoulouMass"]
    # )
    # if total_mass != 1.0:
    #     raise ValueError(f"Total mass must 1.0, not {total_mass}.")

    # The excision surface in the ID grid is distorted to one of constant
    # Boyer-Lindquist radius, meaning that it looks like a Kerr BH. It is not
    # distorted to match the shape of the AH. However, in the Ev grid, we *do*
    # want the excision to match the AH shape so that the control system has an
    # easier time adjusting. Therefore, in order to ensure that the Ev grid lies
    # entirely inside the ID grid regardless of the excision shapes, we make the
    # Ev excision radius a tad larger than the ID excision radius to account for
    # these different excision shapes. This factor was found empirically.
    excision_factor = 1.02

    params = {
        # Initial data files
        "IdFileGlob": str(
            Path(id_run_dir).resolve()
            / (id_input_file["Observers"]["VolumeFileName"] + "*.h5")
        ),
        # Domain geometry
        "ExcisionRadiusA": (
            excision_factor * id_domain_creator["ObjectA"]["InnerRadius"]
        ),
        "ExcisionRadiusB": (
            excision_factor * id_domain_creator["ObjectB"]["InnerRadius"]
        ),
        "XCoordA": id_domain_creator["ObjectA"]["XCoord"],
        "XCoordB": id_domain_creator["ObjectB"]["XCoord"],
        # Initial functions of time
        "InitialAngularVelocity": id_binary["AngularVelocity"],
        "RadialExpansionVelocity": float(id_binary["Expansion"]),
        "HorizonsFile": str(horizons_filename.resolve()),
        "AhASubfileName": "AhA/Coefficients",
        "AhBSubfileName": "AhB/Coefficients",
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }

    # Constraint damping parameters
    params.update(
        _constraint_damping_params(
            mass_left=mass_left,
            mass_right=mass_right,
            initial_separation=initial_separation,
        )
    )

    # Control system
    params.update(
        _control_system_params(
            mass_left=mass_left,
            mass_right=mass_right,
            spin_magnitude_left=spin_magnitude_left,
            spin_magnitude_right=spin_magnitude_right,
        )
    )

    return params


def _load_spec_id_params(id_params_file: Path) -> dict:
    """Load SpEC initial data parameters from 'ID_Params.perl'."""
    # Here we have to deal with SpEC storing the initial data parameters in a
    # perl script (yay!). We convert the perl syntax to YAML and load that into
    # a Python dictionary.
    id_params_yaml = (
        # Drop last three lines that contain the 'ID_Origin' variable
        "\n".join(id_params_file.read_text().split("\n")[:-3])
        # Convert perl syntax to YAML
        .replace("$", "")
        .replace("@", "")
        .replace(";", "")
        .replace(" = ", ": ")
        .replace("=", ": ")
        .replace("(", "[")
        .replace(")", "]")
    )
    logger.debug(f"ID_Params.perl converted to YAML:\n{id_params_yaml}")
    return yaml.safe_load(id_params_yaml)


def inspiral_parameters_spec(
    id_params: dict,
    id_run_dir: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine inspiral parameters from SpEC initial data.

    These parameters fill the 'INSPIRAL_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_params: Initial data parameters loaded from 'ID_Params.perl'.
      id_run_dir: Directory of the initial data, which contains
        'ID_Params.perl' and 'GrDomain.input'.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """

    mass_left = id_params["ID_MB"]
    mass_right = id_params["ID_MA"]
    spin_magnitude_left = id_params["ID_chiBMagnitude"]
    spin_magnitude_right = id_params["ID_chiAMagnitude"]
    initial_separation = id_params["ID_d"]

    params = {
        # Initial data files
        "SpecDataDirectory": str(Path(id_run_dir).resolve()),
        # Domain geometry
        "ExcisionRadiusA": id_params["ID_rExcA"],
        "ExcisionRadiusB": id_params["ID_rExcB"],
        # Off-axis excisions are not supported yet
        "XCoordA": id_params["ID_cA"][0],
        "XCoordB": id_params["ID_cB"][0],
        # Initial functions of time
        "InitialAngularVelocity": id_params["ID_Omega0"],
        "RadialExpansionVelocity": id_params["ID_adot0"],
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }

    # Constraint damping parameters
    params.update(
        _constraint_damping_params(
            mass_left=mass_left,
            mass_right=mass_right,
            initial_separation=initial_separation,
        )
    )

    # Control system
    params.update(
        _control_system_params(
            mass_left=mass_left,
            mass_right=mass_right,
            spin_magnitude_left=spin_magnitude_left,
            spin_magnitude_right=spin_magnitude_right,
        )
    )

    return params


def start_inspiral(
    id_input_file_path: Union[str, Path],
    refinement_level: int = 1,
    polynomial_order: int = 9,
    id_run_dir: Optional[Union[str, Path]] = None,
    inspiral_input_file_template: Union[
        str, Path
    ] = INSPIRAL_INPUT_FILE_TEMPLATE,
    id_horizons_path: Optional[Union[str, Path]] = None,
    continue_with_ringdown: bool = False,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Schedule an inspiral simulation from initial data.

    Point the ID_INPUT_FILE_PATH to the input file of your initial data run,
    or to an 'ID_Params.perl' file from SpEC.
    Also specify 'id_run_dir' if the initial data was run in a different
    directory than where the input file is. Parameters for the inspiral will be
    determined from the initial data and inserted into the
    'inspiral_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    ## Resource allocation

    Runs on 4 nodes by default when scheduled on a cluster. Set 'num_nodes' to
    adjust.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Determine inspiral parameters from initial data
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    if Path(id_input_file_path).name == "ID_Params.perl":
        # Load SpEC initial data (ID_Params.perl)
        inspiral_params = inspiral_parameters_spec(
            _load_spec_id_params(Path(id_input_file_path)),
            id_run_dir,
            refinement_level=refinement_level,
            polynomial_order=polynomial_order,
        )
    else:
        # Load SpECTRE initial data
        with open(id_input_file_path, "r") as open_input_file:
            _, id_input_file = yaml.safe_load_all(open_input_file)
        inspiral_params = inspiral_parameters(
            id_input_file,
            id_run_dir,
            id_horizons_path=id_horizons_path,
            refinement_level=refinement_level,
            polynomial_order=polynomial_order,
        )
    logger.debug(f"Inspiral parameters: {pretty_repr(inspiral_params)}")

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if continue_with_ringdown:
        assert pipeline_dir is not None, (
            "Specify a '--pipeline-dir' / '-d' to continue with the ringdown"
            " simulation automatically. Don't specify a '--run-dir' / '-o' or"
            " '--segments-dir' / '-O' because it will be created in the"
            " 'pipeline_dir' automatically."
        )
        assert run_dir is None and segments_dir is None, (
            "Specify the '--pipeline-dir' / '-d' rather than '--run-dir' / '-o'"
            " or '--segments-dir' / '-O' when continuing with the ringdown"
            " simulation. Directories for the evolution will be created in the"
            " 'pipeline_dir' automatically."
        )
    if pipeline_dir and not run_dir and not segments_dir:
        segments_dir = pipeline_dir / "002_Inspiral"

    # Determine resource allocation
    if (
        scheduler_kwargs.get("scheduler") is not None
        and scheduler_kwargs.get("num_procs") is None
        and scheduler_kwargs.get("num_nodes") is None
    ):
        # Just run on 4 nodes for now, because 1 is surely not enough. We can
        # make this smarter later (e.g. scale with the number of elements).
        scheduler_kwargs["num_nodes"] = 4

    # Schedule!
    return schedule(
        inspiral_input_file_template,
        **inspiral_params,
        **scheduler_kwargs,
        continue_with_ringdown=continue_with_ringdown,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="start-inspiral", help=start_inspiral.__doc__)
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
    "--inspiral-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=INSPIRAL_INPUT_FILE_TEMPLATE,
    help="Input file template for the inspiral.",
    show_default=True,
)
@click.option(
    "--id-horizons-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    show_default="Horizons.h5 inside 'id-run-dir'",
    help=(
        "H5 file that holds information of the horizons of the ID solve. If"
        " this file does not exist in your ID directory, run 'spectre bbh"
        " postprocess-id' in the ID directory to generate it. Note that this is"
        " not needed if you are starting from a SpEC ID_Params.perl file."
    ),
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=1,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=9,
    show_default=True,
)
@click.option(
    "--continue-with-ringdown",
    is_flag=True,
    help=(
        "Continue with the ringdown simulation once a common horizon has"
        " formed."
    ),
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
def start_inspiral_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    start_inspiral(**kwargs)


if __name__ == "__main__":
    start_inspiral_command(help_option_names=["-h", "--help"])
