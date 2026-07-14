# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import click

from spectre.IO.H5.CombineH5Dat import combine_h5_dat
from spectre.support.DirectoryStructure import PipelineStep, list_pipeline_steps
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

CCE_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Cce.yaml"


def run_cce(
    bondi_sachs_data: Union[Union[str, Path], Sequence[Union[str, Path]]],
    force: bool = False,
    cce_input_file_template: Union[str, Path] = CCE_INPUT_FILE_TEMPLATE,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Extract partial or full waveforms from a simulation.

    Pass the files containing Bondi-Sachs data as the first argument. It can be
    a single file, or multiple files that will be combined before running CCE
    (e.g. from multiple segments across inspiral/ringdown). Here, it's important
    that the filename of the BondiSachs data is in the form NameOfFileRXXXX.h5,
    with the last 4 characters being the wave extraction radius. The remaining
    options are forwarded to the 'schedule' command. See 'schedule' docs for
    details.

    Arguments:
        bondi_sachs_data: Path to one or more files containing Bondi-Sachs data
        pipeline_dir: Directory where steps in the pipeline are created.
        run_dir: Directory where the CCE executable is run. Mutually exclusive
        with 'pipeline_dir'.
        cce_input_file_template: Input file template for CCE. This should be a
        yaml file that defines the steps in the CCE pipeline.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the generated"
        " input files."
    )

    if not any([pipeline_dir, run_dir]):
        raise ValueError(
            "Specify either '--run-dir' / '-o' or '--pipeline-dir' / '-d'."
        )
    # If there is a pipeline directory, set run directory as well
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if segments_dir:
        raise ValueError(
            "CCE does not use segments at the moment. Specify"
            " '--run-dir' / '-o' or '--pipeline-dir' / '-d' instead."
        )
    if pipeline_dir and not run_dir:
        pipeline_steps = list_pipeline_steps(pipeline_dir)
        if pipeline_steps:  # Check if the list is not empty
            run_dir = pipeline_steps[-1].next(label="Cce").path
        else:
            run_dir = PipelineStep.first(
                directory=pipeline_dir, label="Cce"
            ).path

    # Check number of arguments. If one, check file ends with extraction radius.
    # For multiple files, ensure all extraction radii are the same and combine
    # for CCE to read a single BondiSachs file.
    if isinstance(bondi_sachs_data, (str, Path)):
        bondi_sachs_data = [bondi_sachs_data]
    bondi_sachs_file = str(Path(bondi_sachs_data[0]).resolve())
    radius_pattern = re.compile(r"R(\d{4})\.h5$")
    match = re.search(radius_pattern, bondi_sachs_file)
    if not match:
        raise ValueError(
            "The provided BondiSachs file does not end with 'RXXXX.h5'."
            " Modify the filename to include the extraction radius in the"
            " format 'NameOfFileRXXXX.h5'. For example, if the extraction"
            " radius is 200, the filename should end with 'R0200.h5'."
        )
    if len(bondi_sachs_data) > 1:
        logger.info("Combining BondiSachs data.")
        extraction_radius = int(match.group(1))
        for path in bondi_sachs_data:
            match = re.search(radius_pattern, str(path))
            if not match or int(match.group(1)) != extraction_radius:
                raise ValueError(
                    "Contradicting extraction radii for files specified. Ensure"
                    " all BondiSachs files end with the same extraction radius."
                )

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        bondi_sachs_file = (
            Path(run_dir).resolve()
            / f"combinedBondiSachsCceR{extraction_radius:04d}.h5"
        )
        combine_h5_dat(
            h5files=bondi_sachs_data,
            output=str(bondi_sachs_file),
            force=force,
            remove_overlapping_segments=True,
        )

    # Create a dictionary of the input parameters for the CCE pipeline. This
    # will be passed to the steps in the pipeline.
    cce_params = {
        "BondiSachsData": bondi_sachs_file,
        "FilesCombined": "\n".join(
            str(Path(path).resolve()) for path in bondi_sachs_data
        ),
    }

    # Determine resource allocation
    if (
        scheduler_kwargs.get("scheduler") is not None
        and scheduler_kwargs.get("num_procs") is None
        and scheduler_kwargs.get("num_nodes") is None
    ):
        # CCE runs best on a single core
        scheduler_kwargs["num_procs"] = 1

    if scheduler_kwargs.get("num_nodes", 1) != 1:
        logger.warning(
            "Forcing number of nodes to 1 for CCE, since CCE does not scale to"
            " more than 1 node."
        )
        scheduler_kwargs["num_nodes"] = 1

    # Schedule!
    return schedule(
        cce_input_file_template,
        **cce_params,
        **scheduler_kwargs,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="run-cce", help=run_cce.__doc__)
@click.argument(
    "bondi_sachs_data",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    nargs=-1,
    required=True,
)
@click.option(
    "--cce-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=CCE_INPUT_FILE_TEMPLATE,
    help="Input file template for CCE.",
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
def run_cce_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    run_cce(**kwargs)


if __name__ == "__main__":
    run_cce_command(help_option_names=["-h", "--help"])
