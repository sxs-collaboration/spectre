# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.IO.H5 as spectre_h5
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.support.DirectoryStructure import PipelineStep, list_pipeline_steps
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.OpenVolfiles import open_volfiles
from spectre.Visualization.ReadH5 import available_subfiles, select_observation

logger = logging.getLogger(__name__)

INSPIRAL_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Inspiral.yaml"

# Resolution levels defined in terms of p-refinement
# To be replaced once AMR is used.
INSPIRAL_LEVS = {
    lev_number: {
        "label": f"Lev{lev_number}",
        "refinement_level": 1,
        "polynomial_order": 7 + lev_number,
    }
    for lev_number in range(-2, 11)
}


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
    min_kinematic_timescale = 1e-2

    size_b_timescale = damping_time_base * 0.2 * mass_left
    # Found that changing the size_a timescale to be the same timescale as
    # size_b fixed early incoming char speeds for higher mass ratio runs.
    size_a_timescale = (
        size_b_timescale
        if (mass_ratio > 2.0)
        else damping_time_base * 0.2 * mass_right
    )

    return {
        "MinKinematicTimescale": min_kinematic_timescale,
        "MaxDampingTimescale": max_damping_timescale,
        "KinematicTimescale": kinematic_timescale,
        "MinSkewTimescale": 5.0 * min_kinematic_timescale,
        "SkewTimescale": 0.5 * (
            5.0 * min_kinematic_timescale + max_damping_timescale
        ),
        "SizeATimescale": size_a_timescale,
        "SizeBTimescale": size_b_timescale,
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
        # SpEC chooses a Gamma0Constant of 0.001, but we found 0.01 produces
        # smaller constraints violations in the envelope/outer shell region.
        "Gamma0Constant": 0.01 / total_mass,
        "Gamma0LeftAmplitude": 4.0 / mass_left,
        "Gamma0LeftWidth": 7.0 * mass_left,
        "Gamma0RightAmplitude": 4.0 / mass_right,
        "Gamma0RightWidth": 7.0 * mass_right,
        # SpEC chooses a Gamm0OriginAmplitude of 0.075, but we found that 0.75
        # produces a smaller burst of constraints from junk radiation.
        "Gamma0OriginAmplitude": 0.75 / total_mass,
        "Gamma0OriginWidth": 2.5 * initial_separation,
        "Gamma1Width": 10.0 * initial_separation,
    }


def inspiral_parameters(
    id_input_file: dict,
    id_metadata: dict,
    id_run_dir: Union[str, Path],
    id_subfile_name: Optional[str],
    id_horizons_path: Optional[Union[str, Path]],
) -> dict:
    """Determine inspiral parameters from SpECTRE initial data.

    These parameters fill the 'INSPIRAL_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_input_file: Initial data input file as a dictionary.
      id_metadata: Metadata of the initial data input file as a dictionary.
      id_run_dir: Directory of the initial data run. Paths in the input file
        are relative to this directory.
      id_subfile_name: Subfile name inside the H5 volume files where the
        initial data is stored. If 'None' and only one subfile exists, that
        subfile is used. Otherwise, the subfile must be specified.
      id_horizons_path: Path to H5 file containing information about the
        horizons in the ID (e.g. mass, spin, spherical harmonic coefficients).
        If this is 'None', the default is the 'Horizons.h5' file inside
        'id_run_dir'.
    """
    # For constraints and control system params we just use the target masses
    # and spins, not the values measured on the horizons, because these numbers
    # don't have to be exact and the horizon quantities will change a bit during
    # the evolution anyway.
    if "TargetParams" in id_metadata:
        target_params = id_metadata["TargetParams"]
    else:
        # Fall back to retrieving the target parameters from the input file
        # for older initial data that didn't store them in the metadata.
        # These are the conformal (background) Kerr-Schild parameters, but they
        # are close enough to the target parameters for our purposes here.
        id_binary = id_input_file["Background"]["Binary"]
        object_left = id_binary["ObjectLeft"]["KerrSchild"]
        object_right = id_binary["ObjectRight"]["KerrSchild"]
        mass_ratio = object_right["Mass"] / object_left["Mass"]
        target_params = {
            "MassRatio": mass_ratio,
            "MassA": object_right["Mass"],
            "MassB": object_left["Mass"],
            "DimensionlessSpinA": object_right["Spin"],
            "DimensionlessSpinB": object_left["Spin"],
        }

    mass_ratio = target_params["MassRatio"]
    mass_a = mass_ratio / (1.0 + mass_ratio)
    mass_b = 1.0 / (1.0 + mass_ratio)
    spin_magnitude_a = np.linalg.norm(target_params["DimensionlessSpinA"])
    spin_magnitude_b = np.linalg.norm(target_params["DimensionlessSpinB"])
    # Initial data can be either from an ID solve or from a previous evolution
    id_from_evolution = "Evolution" in id_input_file
    id_domain_creator = id_input_file["DomainCreator"]["BinaryCompactObject"]
    # This factor is to account for the ID excision not being the same shape as
    # the final horizon found after the last iteration. Found through trial and
    # error that increasing the excision size by this factor allowed the runs to
    # evolve without early incoming char speeds.
    # For unequal masses, this was found through trial and error that
    # decreasing the excision size of object b allowed the runs to evolve
    # without early incoming char speeds.
    excision_radius_factor_a = 1.0 if id_from_evolution else 1.0385
    excision_radius_factor_b = (
        1.0 if (id_from_evolution or mass_ratio > 2.0) else 1.0385
    )
    initial_separation = (
        id_domain_creator["ObjectA"]["XCoord"]
        - id_domain_creator["ObjectB"]["XCoord"]
    )

    # Resolve subfile name in the H5 files
    id_file_glob = str(
        Path(id_run_dir).resolve()
        / (id_input_file["Observers"]["VolumeFileName"] + "*.h5")
    )
    if not id_subfile_name:
        id_subfiles = available_subfiles(
            glob.glob(id_file_glob),
            extension=".vol",
        )
        if len(id_subfiles) == 1:
            id_subfile_name = subfiles[0]
            logger.info(
                f"Selected subfile {id_subfile_name} (the only available one)."
            )
        else:
            raise RequiredChoiceError(
                (
                    "Specify '--id-subfile-name' to select a subfile containing"
                    " volume data."
                ),
                choices=id_subfiles,
            )

    params = {
        # Initial data files
        "IdFileGlob": id_file_glob,
        "IdSubfile": id_subfile_name,
        "IdFromEvolution": id_from_evolution,
        # Domain geometry
        "ExcisionRadiusA": (
            id_domain_creator["ObjectA"]["InnerRadius"]
            * excision_radius_factor_a
        ),
        "ExcisionRadiusB": (
            id_domain_creator["ObjectB"]["InnerRadius"]
            * excision_radius_factor_b
        ),
        "ObjectOuterRadius": initial_separation / 2.5,
        "XCoordA": id_domain_creator["ObjectA"]["XCoord"],
        "XCoordB": id_domain_creator["ObjectB"]["XCoord"],
        "CenterOfMassOffset_y": id_domain_creator["CenterOfMassOffset"][0],
        "CenterOfMassOffset_z": id_domain_creator["CenterOfMassOffset"][1],
        "EnvelopeRadius": 100.0 / 15.0 * initial_separation,
        # SpEC chooses the outer radius based on a Newtonian estimate of the
        # wave zone (See function AutoRmax in SpEC/Support/Perl/SpEC.pm). This
        # may need to be ported over eventually. The CCE extraction radii may
        # also need to be adjusted to account for different outer shell radii.
        "OuterShellRadius": 600.0 / 15.0 * initial_separation,
        # Extra resolution for unequal masses (to be replaced with AMR)
        # This extra refinement was found through trial and error and allowed
        # mass ratio 6 to evolve through inspiral stably.
        "ExtraRadRef": round(mass_ratio / 2.0) - 1 if (mass_ratio > 2.0) else 0,
        "ExtraRadPoints": round(mass_ratio / 5.0) if (mass_ratio > 5.0) else 0,
    }

    # Initial functions of time (set from ID or load from evolution data)
    if id_from_evolution:
        first_volfile = id_file_glob.replace("*", "0")
        _, obs_time = select_observation(
            open_volfiles([first_volfile], id_subfile_name), step=-1
        )
        params.update(
            {
                "InitialTime": obs_time,
                "FotFilename": first_volfile,
            }
        )
    else:
        id_shape_A = id_domain_creator["TimeDependentMaps"]["ShapeMapA"]
        id_shape_B = id_domain_creator["TimeDependentMaps"]["ShapeMapB"]
        id_binary = id_input_file["Background"]["Binary"]
        horizons_filename = (
            Path(id_horizons_path)
            if id_horizons_path is not None
            else Path(id_run_dir) / "Horizons.h5"
        )
        # Initial functions of time
        params.update(
            {
                "InitialAngularVelocity": id_binary["AngularVelocity"],
                "RadialExpansionVelocity": float(id_binary["Expansion"]),
                "HorizonsFile": str(horizons_filename.resolve()),
                "AhASubfileName": "AhA/Coefficients",
                "AhBSubfileName": "AhB/Coefficients",
                "ExcisionAShapeMass": id_shape_A["InitialValues"]["Mass"],
                "ExcisionAShapeSpin_x": id_shape_A["InitialValues"]["Spin"][0],
                "ExcisionAShapeSpin_y": id_shape_A["InitialValues"]["Spin"][1],
                "ExcisionAShapeSpin_z": id_shape_A["InitialValues"]["Spin"][2],
                "ExcisionBShapeMass": id_shape_B["InitialValues"]["Mass"],
                "ExcisionBShapeSpin_x": id_shape_B["InitialValues"]["Spin"][0],
                "ExcisionBShapeSpin_y": id_shape_B["InitialValues"]["Spin"][1],
                "ExcisionBShapeSpin_z": id_shape_B["InitialValues"]["Spin"][2],
            }
        )

    # Constraint damping parameters
    params.update(
        _constraint_damping_params(
            mass_left=mass_b,
            mass_right=mass_a,
            initial_separation=initial_separation,
        )
    )

    # Control system
    params.update(
        _control_system_params(
            mass_left=mass_b,
            mass_right=mass_a,
            spin_magnitude_left=spin_magnitude_b,
            spin_magnitude_right=spin_magnitude_a,
        )
    )

    # Store target parameters in the input file
    params["TargetParams"] = yaml.safe_dump(
        {"TargetParams": target_params}
    ).strip()

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
) -> dict:
    """Determine inspiral parameters from SpEC initial data.

    These parameters fill the 'INSPIRAL_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_params: Initial data parameters loaded from 'ID_Params.perl'.
      id_run_dir: Directory of the initial data, which contains
        'ID_Params.perl' and 'GrDomain.input'.
    """

    mass_left = id_params["ID_MB"]
    mass_right = id_params["ID_MA"]
    mass_ratio = mass_right / mass_left
    spin_magnitude_left = id_params["ID_chiBMagnitude"]
    spin_magnitude_right = id_params["ID_chiAMagnitude"]
    initial_separation = id_params["ID_d"]

    params = {
        # Initial data files
        "SpecDataDirectory": str(Path(id_run_dir).resolve()),
        # Domain geometry
        # SpEC excision in ID_Params.perl is 0.89 * horizon radius, but
        # usually you want to excise less than the maximum. Here use 6% larger,
        # or about 0.9434 * horizon radius.
        "ExcisionRadiusA": id_params["ID_rExcA"] * 1.06,
        "ExcisionRadiusB": id_params["ID_rExcB"] * 1.06,
        "ObjectOuterRadius": initial_separation / 2.5,
        "XCoordA": id_params["ID_cA"][0],
        "XCoordB": id_params["ID_cB"][0],
        # COM offset in y and z is the same for both objects
        "CenterOfMassOffset_y": id_params["ID_cA"][1],
        "CenterOfMassOffset_z": id_params["ID_cA"][2],
        # Initial functions of time
        "InitialAngularVelocity": id_params["ID_Omega0"],
        "RadialExpansionVelocity": id_params["ID_adot0"],
        "EnvelopeRadius": 100.0 / 15.0 * initial_separation,
        # SpEC chooses the outer radius based on a Newtonian estimate of the
        # wave zone (See function AutoRmax in SpEC/Support/Perl/SpEC.pm). This
        # may need to be ported over eventually. The CCE extraction radii may
        # also need to be adjusted to account for different outer shell radii.
        "OuterShellRadius": 600.0 / 15.0 * initial_separation,
        # Extra resolution for unequal masses (to be replaced with AMR)
        # This extra refinement was found through trial and error and allowed
        # mass ratio 6 to evolve through inspiral stably.
        "ExtraRadRef": round(mass_ratio / 2.0) - 1 if (mass_ratio > 2.0) else 0,
        "ExtraRadPoints": round(mass_ratio / 5.0) if (mass_ratio > 5.0) else 0,
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
    lev: Optional[int] = None,
    refinement_level: Optional[int] = None,
    polynomial_order: Optional[int] = None,
    id_run_dir: Optional[Union[str, Path]] = None,
    id_subfile_name: Optional[str] = None,
    inspiral_input_file_template: Union[
        str, Path
    ] = INSPIRAL_INPUT_FILE_TEMPLATE,
    id_horizons_path: Optional[Union[str, Path]] = None,
    continue_with_ringdown: bool = False,
    eccentricity_control: bool = False,
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
    assert not (continue_with_ringdown and eccentricity_control), (
        "Cannot enable both 'continue_with_ringdown' and"
        " 'eccentricity_control'. Choose which of the two to perform next."
    )

    # Determine inspiral parameters from initial data
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    if Path(id_input_file_path).name == "ID_Params.perl":
        # Load SpEC initial data (ID_Params.perl)
        inspiral_params = inspiral_parameters_spec(
            _load_spec_id_params(Path(id_input_file_path)),
            id_run_dir,
        )
    else:
        # Load SpECTRE initial data
        with open(id_input_file_path, "r") as open_input_file:
            id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
        inspiral_params = inspiral_parameters(
            id_input_file,
            id_metadata,
            id_run_dir,
            id_subfile_name=id_subfile_name,
            id_horizons_path=id_horizons_path,
        )

    # Determine resolution
    if lev is not None:
        assert (refinement_level is None) and (polynomial_order is None), (
            "The option 'lev' is mutaully exclusive with 'refinement_level' and"
            " 'polynomial_order'."
        )
        selected_lev = INSPIRAL_LEVS[lev]
        refinement_level = selected_lev["refinement_level"]
        polynomial_order = selected_lev["polynomial_order"]
    else:
        assert (refinement_level is not None) and (
            polynomial_order is not None
        ), (
            "Resolution not specified. Provide either 'lev' or both"
            " 'refinement_level' and 'polynomial_order'."
        )
    inspiral_params.update(
        {
            "Lev": lev,
            "L": refinement_level,
            "P": polynomial_order,
        }
    )

    # Set final time for eccentricity control to 2-3 orbits. This can be set
    # more dynamically in the future.
    if eccentricity_control:
        inspiral_params["FinalTime"] = (
            500 + 5 * np.pi / inspiral_params["InitialAngularVelocity"]
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
        pipeline_steps = list_pipeline_steps(pipeline_dir)
        if pipeline_steps:  # Check if the list is not empty
            segments_dir = pipeline_steps[-1].next(label="Inspiral").path
        else:
            segments_dir = PipelineStep.first(
                directory=pipeline_dir, label="Inspiral"
            ).path

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
        eccentricity_control=eccentricity_control,
        id_input_file_path=Path(id_input_file_path).resolve(),
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
    "--id-subfile-name",
    help=(
        "Name of the subfile within the initial data H5 files containing the "
        "volume data to read in. "
        "Optional if the H5 files have only one subfile."
    ),
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
    "--lev",
    type=int,
    help=(
        "Resolution levels defined in terms of h and p refinement. Can be"
        " specified multiple times. For integer N, LevN corresponds to"
        " refinement level L = 1 and polynomial order P = 7 + N. Mutually"
        " exclusive with options '-L' and '-P'"
    ),
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
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
    "--eccentricity-control",
    is_flag=True,
    help=(
        "Perform eccentricity reduction script that finds current eccentricity"
        "and better guesses for the input orbital parameters."
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
