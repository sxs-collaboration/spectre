# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import jinja2
import numpy as np
import yaml
from rich.pretty import pretty_repr

from spectre.support.BinDirectory import (
    BIN_DIR_NAME,
    SUBMIT_SCRIPT_TEMPLATE,
    BinDirectory,
)
from spectre.support.DirectoryStructure import (
    Checkpoint,
    Segment,
    list_checkpoints,
    list_segments,
)
from spectre.support.Machines import this_machine
from spectre.support.RunNext import run_next
from spectre.support.Yaml import SafeDumper
from spectre.tools.ValidateInputFile import validate_input_file
from spectre.Visualization.ReadInputFile import find_phase_change

logger = logging.getLogger(__name__)
machine = this_machine(raise_exception=False)


def _resolve_executable(
    executable: Union[str, Path], bin_dir: Optional[BinDirectory] = None
) -> Path:
    """The absolute path of the 'executable'

    Looks in the 'bin_dir' first, if one is given, and then in the 'PATH'.
    Raises 'ValueError' if the executable is not found.

    The 'bin_dir' comes first because a simulation that has an executable
    frozen runs that one, whatever the environment offers.
    """
    logger.debug(f"Resolving executable: {executable}")
    if bin_dir:
        in_bin_dir = bin_dir.executable(executable)
        if in_bin_dir.is_file():
            return in_bin_dir
    # This default bin dir is already added in spectre.__main__.py, but only
    # when running the CLI. It is the bin dir of the build directory that
    # contains this script, or the simulation's bin directory when running from
    # one. When running Python code outside the CLI this should also be the
    # default bin dir.
    path = os.environ["PATH"] + ":" + str(BinDirectory.this().path.resolve())
    which_exec = shutil.which(executable, path=path)
    if which_exec:
        return Path(which_exec).resolve()
    raise ValueError(
        f"Executable not found: {executable}. Make sure it is compiled. To"
        " look for executables in a specific build directory make sure it"
        " is in the 'PATH' or use the 'spectre --build-dir / -b' option."
    )


def _write_or_overwrite(
    text: str, path: Path, error_hint: Optional[str] = None, force: bool = False
):
    """Write the 'text' to a file at 'path'

    Raise an 'OSError' if the file already exists, unless called with 'force'
    or if the existing file content is identical to the 'text'.
    The 'hint' is appended to the error message.
    """
    if path.exists():
        if path.read_text() == text:
            return
        if not force:
            raise OSError(
                f"File already exists at '{path}'. Retry with "
                "'force' ('--force' / '-f') to overwrite."
                + (("\n" + error_hint) if error_hint else "")
            )
    logger.debug(f"Write file: {path}")
    path.write_text(text)


def schedule(
    input_file_template: Union[str, Path],
    scheduler: Optional[Union[str, Sequence]],
    no_schedule: Optional[bool] = None,
    executable: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    create_bin: Optional[bool] = None,
    bin_dir: Optional[Union[str, Path]] = None,
    copy_extra_executables: Optional[Sequence[Union[str, Path]]] = None,
    job_name: Optional[str] = None,
    submit_script_template: Optional[Union[str, Path]] = None,
    from_checkpoint: Optional[Union[str, Path, Checkpoint]] = None,
    input_file_name: Optional[str] = None,
    submit_script_name: str = "Submit.sh",
    out_file_name: str = "spectre.out",
    context_file_name: str = "SchedulerContext.yaml",
    submit: Optional[bool] = None,
    clean_output: bool = False,
    force: bool = False,
    validate: Optional[bool] = True,
    profile_with: Optional[str] = None,
    extra_params: dict = {},
    **kwargs,
) -> Optional[subprocess.CompletedProcess]:
    """Schedule executable runs with an input file

    Configures the input file, submit script, etc. to the 'run_dir', and then
    invokes the 'scheduler' to submit the run (typically "sbatch"). You can also
    bypass the scheduler and run the executable directly by setting the
    'scheduler' to 'None'.

    # Selecting the executable

    Specify either a path to the executable, or just its name if it's in the
    'PATH'. If unspecified, the 'Executable' listed in the input file metadata
    is used.

    # The bin directory

    By default the scheduled job runs from a 'bin' directory of the simulation
    instead of from the build directory, so that recompiling, switching
    branches or deleting the build directory can't change or break queued jobs,
    later segments, or later pipeline steps.

    Which bin directory a run uses is decided by looking for it: the search
    starts at the 'run_dir' and works outwards, and the nearest hit wins (see
    'BinDirectory.find'). Runs in subdirectories of a simulation (e.g.
    eccentricity-control iterations, resolution branches) therefore share its
    bin directory rather than creating their own. The search never leaves the
    simulation, so a run tree placed below an unrelated simulation does not pick
    up that simulation's bin directory. A build directory's 'bin' is a hit like
    any other: a run tree sitting directly in a build directory runs from it,
    and nothing is copied.

    Only when the search finds none is one created, and then for the simulation
    as a whole rather than for this run alone: in the 'pipeline_dir' if there
    is one, so all steps of a pipeline share it, otherwise in the
    'segments_dir', otherwise in the 'run_dir'. Its path is recorded in the
    scheduler context, so later segments find it again. It is created once and
    never updated implicitly: a file already in it is kept rather than
    replaced, so moving a running simulation onto new code means replacing
    files there by hand, at your own risk.
    Note that creating a bin directory fails if the build is configured with
    'BUILD_SHARED_LIBS=ON', because such executables load shared libraries out
    of the build directory and would stop working when it changes.

    The default submit script template is one of the support files next to the
    bin directory. If you want to change how later segments or pipeline steps
    are submitted, edit '<simulation>/support/SubmitTemplate.sh'. If you provide
    a different submit script template (through `--submit-script-template`), its
    path is passed on to later segments, but it is not copied into the bin
    directory, so it must remain available at the same path for later segments
    to find it.

    An executable is looked up in the bin directory before the 'PATH': that is
    what makes a simulation run the same binary throughout. An executable that
    the bin directory does not hold yet is resolved in the environment and
    copied in. To freeze all pipeline executables at the start, provide them to
    'copy_extra_executables'.

    Running the executable directly (no 'scheduler') does not create a bin
    directory by default, because nothing runs unsupervised afterwards, but
    'create_bin' still creates one when it is set explicitly. A direct run does
    use a bin directory that already exists, and adds its executable to it, so
    it runs the same copy the scheduled jobs of that simulation do.

    # Segments and run directories

    You can set either the 'run_dir' or the 'segments_dir' to specify where the
    executable will run (but not both). If you specify a 'run_dir', the
    executable will run in it directly. If you specify a 'segments_dir', a new
    segment will be created and used as the 'run_dir'. Segments are named with
    incrementing integers and continue the run from the previous segment. For
    example, the following is a typical 'segments_dir':

    \b
    ```sh
    # Everything the scheduled jobs run from (see above)
    bin/
        MyExecutable
        spectre
        python/
    # Support files, copied from the build directory
    support/
        Machine.yaml
        SubmitTemplateBase.sh
        SubmitTemplate.sh
    # One segment per day
    Segment_0000/
        InputFile.yaml
        Submit.sh
        Output.h5
        # Occasional checkpoints, and a checkpoint before termination
        Checkpoints/
            Checkpoint_0000/
            Checkpoint_0001/...
    # Next segment continues from last checkpoint of previous segment
    Segment_0001/...
    ```

    You can omit the 'run_dir' if the current working directory already contains
    the input file.

    # Placeholders

    The input file, submit script, 'run_dir', 'segments_dir', and 'job_name' can
    have placeholders like '{{ num_nodes }}'. They must conform to the
    [Jinja template format](https://jinja.palletsprojects.com/en/3.0.x/templates/).
    The placeholders are resolved in the following stages.
    The following parameters are available as placeholders:

    1. 'job_name' (if specified):

    \b
        - All arguments to this function, including all additional '**kwargs'.
          For example, the additional '**kwargs' can include parameters
          controlling resolution in the input file.
        - 'executable_name': Just the name of the executable (basename of the
          'executable' as given).

    2. 'run_dir' and 'segments_dir':

    \b
        - All parameters from the previous stage.
        - 'job_name': Either the resolved 'job_name' from the previous stage, or
          the 'executable_name' if unspecified.

    3. Input file & submit script:

    \b
        - All parameters from the previous stages.
        - 'run_dir': Absolute path to the 'run_dir'.
        - 'segments_dir': Absolute path to the 'segments_dir', or 'None' if no
          segments directory is available.
        - 'input_file': Relative path to the configured input file (in the
          'run_dir').
        - 'out_file': Absolute path to the log file (in the 'run_dir').
        - 'executable': Absolute path to the executable that will run.
        - 'spectre_cli': Absolute path to the SpECTRE CLI.
        - Typical additional parameters used in submit scripts are 'queue' and
          'time_limit'.

    The parameters used to render the submit script are stored in a context file
    (named 'context_file_name') in the 'run_dir' to support resubmissions. The
    context file is used by 'spectre.support.resubmit' to schedule the next
    segment using the same parameters.

    # Scheduling multiple runs

    You can pass ranges of parameters to the '**kwargs' of this function to
    schedule multiple runs using the same input file template. For example, you
    can do an h-convergence test by using a placeholder for the refinement level
    in your input file:

    \b
    ```yaml
    # In the domain creator:
    InitialRefinement: {{ lev }}
    ```

    When a parameter in '**kwargs' is an iterable, the 'schedule' function will
    recurse for every element in the iterable. For example, you can schedule
    multiple runs for a convergence test like this:

    \b
    ```py
    schedule(
        run_dir="Lev{{ lev }}",
        # ...
        lev=range(1, 3))
    ```

    \f
    Arguments:
      input_file_template: Path to an input file. It will be copied to the
        'run_dir'. It can be a Jinja template (see above).
      scheduler: 'None' to run the executable directly, or a scheduler such as
        "sbatch" to submit the run to a queue.
      no_schedule: Optional. If 'True', override the 'scheduler' to 'None'.
        Useful to specify on the command line where the 'scheduler' defaults to
        "sbatch" on clusters.
      executable: Path or name of the executable to run. If unspecified, use the
        'Executable' set in the input file metadata.
      run_dir: The directory to which input file, submit script, etc. are
        copied, and relative to which the executable will run.
        Can be a Jinja template (see above).
      segments_dir: The directory in which a new segment is created as the
        'run_dir'. Mutually exclusive with 'run_dir'.
        Can be a Jinja template (see above).
      create_bin: Create a bin directory for the simulation, and run from it.
        By default (when set to 'None'), create one if a 'pipeline_dir' or a
        'segments_dir' is set, i.e. for any run that will be continued, and only
        if a 'scheduler' is used (job submission). When set to 'True', create
        one also for a plain '--run-dir' / '-o' run. When set to 'False', run
        from the build directory. This controls only creation: an existing bin
        directory, here or in an enclosing directory, is used either way.
      bin_dir: Optional. Path to the simulation's bin directory. Defaults to
        'bin' in the 'pipeline_dir', 'segments_dir' or 'run_dir' (in that
        order of preference). Set automatically for later segments, because it
        is recorded in the context file.
      copy_extra_executables: Optional. Additional executables to copy into the
        bin directory, e.g. the executables of later pipeline steps, which
        have to be there before the handoff to them happens. They are resolved
        like the 'executable'.
      job_name: Optional. A string describing the job.
        Can be a Jinja template (see above). (Default: executable name)
      submit_script_template: Optional. Path to a submit script, used as
        described above. Can be a Jinja template (see above). (Default:
        'support/SubmitTemplate.sh' next to the bin directory)
      from_checkpoint: Optional. Path to a checkpoint directory.
      input_file_name: Optional. Filename of the input file in the 'run_dir'.
        (Default: basename of the 'input_file_template')
      submit_script_name: Optional. Filename of the submit script. (Default:
        "Submit.sh")
      out_file_name: Optional. Name of the log file. (Default:
        "spectre.out")
      context_file_name: Optional. Name of the file that stores the context
        for resubmissions in the `run_dir`. Used by `spectre.support.resubmit`.
        (Default: "SchedulerContext.yaml")
      submit: Optional. If 'True', automatically submit jobs using the
        'scheduler'. If 'False', skip the job submission. If 'None', prompt for
        confirmation before submitting.
      clean_output: Optional. When 'True', use
        'spectre.tools.CleanOutput.clean_output' to clean up existing output
        files in the 'run_dir' before scheduling the run. (Default: 'False')
      force: Optional. When 'True', overwrite input file and submit script
        in the 'run_dir' instead of raising an error when they already exist.
      validate: Optional. When 'True', validate that the input file is parsed
        correctly. When 'False' skip this step.
      profile_with: Optional. When set to "hpctoolkit", enable profiling with
        HPCToolkit. No other profilers are currently supported. The executable
        must be compiled so that it's compatible with HPCToolkit (see
        https://spectre-code.org/profiling.html). This will modify the submit
        script to run the executable with 'hpcrun' and postprocess the profiling
        data with 'hpcstruct' and 'hpcprof'. (Default: False).
      extra_params: Optional. Dictionary of extra parameters passed to input
        file and submit script templates. Parameters can also be passed as
        keyword arguments to this function instead.

    Returns: The 'subprocess.CompletedProcess' representing either the process
      that scheduled the run, or the process that ran the executable if
      'scheduler' is 'None'. Returns 'None' if no or multiple runs were
      scheduled.
    """
    # Defaults
    input_file_template = Path(input_file_template)
    if not input_file_name:
        input_file_name = input_file_template.resolve().name
    if no_schedule:
        scheduler = None
    if isinstance(from_checkpoint, Checkpoint):
        from_checkpoint = from_checkpoint.path
    if from_checkpoint:
        from_checkpoint = Path(from_checkpoint).resolve()
    # Snapshot function arguments for template substitutions
    kwargs.update(extra_params)
    del extra_params
    all_args = locals().copy()
    del all_args["kwargs"]

    # Recursively schedule ranges of runs
    for key, value in kwargs.items():
        # Check if the parameter is an iterable
        if isinstance(value, str):
            # Strings are iterable, but we don't want to treat them as such
            continue
        try:
            iter(value)
        except TypeError:
            continue
        # Recurse for each value of the iterable
        for value_i in value:
            logger.info(f"Recurse for {key}={value_i}")
            kwargs_i = kwargs.copy()
            kwargs_i.update({key: value_i})
            try:
                schedule(**all_args, **kwargs_i)
            except:
                logger.exception(f"Recursion for {key}={value_i} failed.")
        return

    # Resolve number of cores, nodes, etc.
    num_procs = kwargs.get("num_procs")
    num_nodes = kwargs.get("num_nodes")
    num_slurm_tasks = None
    if num_procs:
        assert (
            num_nodes is None or num_nodes == 1
        ), "Specify either 'num_procs' or 'num_nodes', not both."
        if machine:
            # Approximately round up to nearest number of full nodes
            cores_per_node = (
                machine.DefaultTasksPerNode * machine.DefaultProcsPerTask
            )
            if num_procs <= cores_per_node:
                num_nodes = 1
                num_slurm_tasks = int(
                    np.ceil(num_procs / machine.DefaultProcsPerTask)
                )
            else:
                num_nodes = int(np.ceil(num_procs / cores_per_node))
                logger.info(f"Rounded up to run on {num_nodes} full node(s).")
                num_procs = None
        else:
            num_nodes = 1
    # Update kwargs with resolved num_procs and num_nodes (used to build
    # `context` below)
    kwargs.update(
        num_procs=num_procs,
        num_nodes=num_nodes,
        num_slurm_tasks=num_slurm_tasks,
    )

    # Set up template environment with basic configuration
    template_env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    # Start collecting parameters for template substitutions. We filter 'None'
    # values so they properly trigger undefined-variable errors when templates
    # need them.
    context = {
        key: value
        for key, value in dict(**all_args, **kwargs).items()
        if value is not None
    }

    # Read input file template
    input_file_contents = input_file_template.read_text()

    # Resolve executable
    if not executable:
        # Can't parse the full input file yet because we haven't collected all
        # parameters yet. Instead, just parse the metadata. We use the YAML
        # document start indicator '---' to drop the rest of the input file.
        # Note that the document start indicator is optional for the first
        # document in the file and there may be comments or a version directive
        # before it, so we drop the last document in the file rather than split
        # on the first '---'.
        metadata_template = input_file_contents.rpartition("---")[0]
        metadata_yaml = template_env.from_string(metadata_template).render(
            context
        )
        metadata = yaml.safe_load(metadata_yaml)
        try:
            executable = metadata["Executable"]
        except (KeyError, TypeError) as err:
            raise ValueError(
                "Specify an 'executable' ('--executable' / '-E') "
                "or list one in the input file metadata "
                "as 'Executable:'."
            ) from err
    # Only the name is needed here, to render the job name and the run
    # directory. Where the executable actually is gets settled further below,
    # because that answer depends on the bin directory, which is looked for at
    # the run directory.
    executable_name = Path(executable).name
    context.update(executable_name=executable_name)

    # Resolve job_name
    if job_name:
        job_name = template_env.from_string(job_name).render(context).strip()
    else:
        job_name = executable_name
    context.update(job_name=job_name)

    # Resolve run_dir and segments_dir
    if run_dir and segments_dir:
        raise ValueError(
            "Specify either 'run_dir' ('--run-dir' / '-o') "
            "or 'segments_dir' ('--segments-dir' / '-O'), not both."
        )
    elif not run_dir and not segments_dir:
        # Neither run_dir nor segments_dir were specified. Set run_dir to the
        # current working directory.
        if input_file_template.resolve().parent == Path.cwd():
            run_dir = Path.cwd()
        else:
            raise ValueError(
                "Specify a 'run_dir' ('--run-dir' / '-o') "
                "or a 'segments_dir' ('--segments-dir' / '-O'), "
                "or place the input file into the current directory."
            )
    # At this point either run_dir or segments_dir are set, but not both.
    if run_dir:
        # Run directly in the run_dir. If the run_dir looks like a segment, set
        # the segments_dir so resubmitting works.
        run_dir = Path(
            template_env.from_string(str(run_dir)).render(context).strip()
        )
        if Segment.match(run_dir):
            segments_dir = run_dir.resolve().parent
            all_segments = list_segments(segments_dir)
    else:
        # Run in next segment in the segments_dir
        segments_dir = Path(
            template_env.from_string(str(segments_dir)).render(context).strip()
        )
        all_segments = list_segments(segments_dir)
        if all_segments:
            run_dir = all_segments[-1].next.path
        else:
            run_dir = Segment.first(segments_dir).path
    if segments_dir and all_segments:
        # Make sure we're continuing the last checkpoint of the last segment.
        # This requirement can be relaxed in the future if needed.
        assert from_checkpoint, (
            f"Found existing segments in directory '{segments_dir}'. Use"
            " '--from-last-checkpoint' to continue from the last"
            " checkpoint in this directory."
        )
        last_segment = all_segments[-1]
        assert (
            from_checkpoint.parent == last_segment.checkpoints_dir.resolve()
        ), (
            "You're not continuing from the last segment"
            f" ({last_segment.path}). It is technically possible to continue"
            " from a different checkpoint, but probably wrong. This assert"
            " safeguards against inconsistent usage. If you want to continue"
            f" from the checkpoint you specified ({from_checkpoint}), choose a"
            " different directory to run in. Otherwise, use"
            " '--from-last-checkpoint SEGMENTS_DIR' to continue from the"
            " latest segment."
        )
        last_segment_checkpoints = last_segment.checkpoints
        assert last_segment_checkpoints, (
            f"The last segment '{last_segment.path}' has no checkpoints to"
            " continue from. It is technically possible to continue from a"
            " different checkpoint, but probably wrong. This assert safeguards"
            " against inconsistent usage. If you want to continue from the"
            f" checkpoint you specified ({from_checkpoint}), choose a different"
            " directory to run in. Otherwise, remove the incomplete segment"
            " and use '--from-last-checkpoint SEGMENTS_DIR' to continue from"
            " the latest segment."
        )
        last_checkpoint = last_segment_checkpoints[-1]
        assert from_checkpoint == last_checkpoint.path.resolve(), (
            "You're not continuing from the previous segment's last checkpoint"
            f" ({last_checkpoint.path}). This is technically possible, but"
            " probably wrong. This assert safeguards against inconsistent"
            " usage. If you want to continue from the checkpoint you specified"
            f" ({from_checkpoint}), choose a different directory to run in."
            " Otherwise, use '--from-last-checkpoint LAST_SEGMENT' to continue"
            " from the last checkpoint."
        )
    context.update(run_dir=run_dir.resolve())
    if segments_dir:
        context.update(segments_dir=segments_dir.resolve())

    # Resolve outfile
    out_file = run_dir / out_file_name
    context.update(out_file=out_file.resolve())

    # Create the run directory
    logger.info(f"Configure run directory '{run_dir}'")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the existing bin directory of the simulation. Pipelines pass a
    # 'pipeline_dir' as an additional parameter, and all steps of a pipeline
    # share one bin directory.
    pipeline_dir = kwargs.get("pipeline_dir")
    if bin_dir:
        bin_dir = BinDirectory(Path(bin_dir).resolve())
        if not bin_dir.path.exists():
            # The recorded path is stale, so the simulation directory was
            # probably moved or renamed. Fall through to the ordinary look-up
            # below.
            logger.warning(
                "The bin directory recorded for this run does not exist:"
                f" '{bin_dir.path}'. Looking for another one."
            )
            bin_dir = None
    if not bin_dir:
        # Look for the simulation's bin directory whether or not there is a
        # scheduler: a run that executes directly still belongs to the
        # simulation and uses its executables. 'create_bin' governs only
        # whether one is created.
        bin_dir = BinDirectory.find(run_dir)
        if bin_dir:
            logger.debug(
                f"Using the bin directory of the simulation: '{bin_dir.path}'"
            )

    # Resolve the executables now that the bin directory is known
    executable = _resolve_executable(executable, bin_dir)
    extra_executables = [
        _resolve_executable(extra_executable, bin_dir)
        for extra_executable in (copy_extra_executables or [])
    ]

    # Create the bin directory if needed
    if create_bin is None:
        create_bin = bool(pipeline_dir or segments_dir) and scheduler
    if not bin_dir and create_bin:
        bin_dir = BinDirectory.create(
            path=(
                Path(pipeline_dir or segments_dir or run_dir) / BIN_DIR_NAME
            ).resolve(),
        )
    # Now the bin dir is fully resolved
    if bin_dir:
        context.update(bin_dir=bin_dir.path)
    context.update(create_bin=create_bin)

    # Copy the executables into the bin dir if they're not already there, and
    # update the executable paths to point to the bin dir
    if bin_dir:
        executable = bin_dir.add(executable)
        for extra_executable in extra_executables:
            bin_dir.add(extra_executable)
    context.update(executable=executable)
    logger.info(f"Running with executable: {executable}")

    # Configure input file
    input_file_path = run_dir / input_file_name
    context.update(input_file=input_file_name)
    logger.debug(
        f"Configure input file template '{input_file_template}' with these"
        f" parameters: {pretty_repr(context)}"
    )
    rendered_input_file = template_env.from_string(input_file_contents).render(
        context
    )
    _write_or_overwrite(
        rendered_input_file,
        input_file_path,
        error_hint=(
            "If you're scheduling multiple runs, use "
            "placeholders in the directory name such as:\n"
            "  -p lev=1...3 --run-dir 'Lev{{ lev }}'"
        ),
        force=force,
    )

    # Validate input file
    if validate:
        validate_input_file(
            input_file_path.resolve(), executable=executable, work_dir=run_dir
        )

    # - If the input file may request resubmissions, make sure we have a
    #   segments directory
    metadata, input_file = yaml.safe_load_all(rendered_input_file)
    wallclock_exit_phase_change = find_phase_change(
        "CheckpointAndExitAfterWallclock", input_file
    )
    if wallclock_exit_phase_change is not None and not segments_dir:
        raise ValueError(
            "Found 'CheckpointAndExitAfterWallclock' in the input file but "
            "no 'segments_dir' ('--segments-dir' / '-O') is set. "
            "Specify a segments directory to enable resubmissions, or "
            "remove 'CheckpointAndExitAfterWallclock' from the input file."
        )

    # Clean output
    if clean_output:
        from spectre.tools.CleanOutput import clean_output

        clean_output(input_file=input_file_path, output_dir=run_dir, force=True)

    # If requested, run executable directly and return early
    if not scheduler:
        assert num_nodes is None or num_nodes == 1, (
            "Running executables directly is only supported on a single node. "
            "Set the 'scheduler' ('--scheduler') to submit a multi-node job "
            "to the queue."
        )
        auto_provision = num_procs is None
        provision_info = (
            "all available cores"
            if auto_provision
            else f"{num_procs} core{'s'[:num_procs!=1]}"
        )
        logger.info(
            f"Run '{executable.name}' in '{run_dir}' on {provision_info}."
        )
        machine = this_machine(raise_exception=False)
        if profile_with is None:
            profiling_command = []
        elif profile_with == "hpctoolkit":
            profiling_command = [
                "hpcrun",
                "-t",
                "-o",
                "hpctoolkit-measurements",
            ]
        else:
            raise ValueError(
                f"Unsupported profiler: {profile_with}. Currently, only"
                " 'hpctoolkit' is supported."
            )
        run_command = (
            (machine.launch_command if machine else [])
            + profiling_command
            + [
                str(executable),
                "--input-file",
                str(input_file_path.resolve()),
            ]
        )
        if auto_provision:
            run_command += ["+auto-provision"]
        else:
            run_command += ["+p", str(num_procs)]
        if from_checkpoint:
            run_command += ["+restart", str(from_checkpoint)]
        logger.debug(f"Run command: {run_command}")
        if submit is False:
            return
        env = os.environ.copy()
        # Disable multithreading so our executables have control over the
        # parallelization
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        process = subprocess.Popen(run_command, cwd=run_dir, env=env)
        # Realtime streaming of _captured_ stdout and stderr to the console
        # doesn't seem to work reliably, so we just let the process stream
        # directly to the console and wait for it to complete.
        process.wait()
        # Raise errors on non-zero exit codes
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=run_command
            )
        if profile_with == "hpctoolkit":
            subprocess.run(
                ["hpcstruct", "hpctoolkit-measurements"],
                cwd=run_dir,
                check=True,
            )
            subprocess.run(
                [
                    "hpcprof",
                    "-o",
                    "hpctoolkit-database",
                    "hpctoolkit-measurements",
                ],
                cwd=run_dir,
                check=True,
            )
        # Run the 'Next' entrypoint listed in the input file metadata
        if metadata and "Next" in metadata:
            run_next(
                metadata["Next"], input_file_path=input_file_path, cwd=run_dir
            )
        return process

    # The CLI that the job calls for resubmissions and handoffs
    spectre_cli = (bin_dir or BinDirectory.this()).spectre_cli
    context.update(spectre_cli=spectre_cli)

    # Configure submit script
    # Render from the simulation's frozen support files, or from the build
    # directory if not available.
    support_dir = (bin_dir or BinDirectory.this()).support_dir.resolve()
    if submit_script_template:
        submit_script_template = Path(submit_script_template).resolve()
    else:
        submit_script_template = support_dir / SUBMIT_SCRIPT_TEMPLATE
    context.update(submit_script_template=submit_script_template)
    logger.debug(
        f"Configure submit script template '{submit_script_template}' with"
        f" these parameters: {pretty_repr(context)}"
    )
    # Use a FileSystemLoader to support template inheritance
    submit_script_template_env = template_env.overlay(
        loader=jinja2.FileSystemLoader(
            [submit_script_template.parent, support_dir]
        )
    )
    rendered_submit_script = submit_script_template_env.get_template(
        submit_script_template.name
    ).render(context)
    submit_script_path = run_dir / submit_script_name
    _write_or_overwrite(rendered_submit_script, submit_script_path, force=force)

    # Write context to file to support resubmissions
    if segments_dir:
        with open(run_dir / context_file_name, "w") as open_context_file:
            yaml.dump(context, open_context_file, Dumper=SafeDumper)

    # Submit
    if submit or (
        submit is None
        and click.confirm(f"Submit '{submit_script_path}'?", default=True)
    ):
        if isinstance(scheduler, str):
            scheduler = [scheduler]
        submit_process = subprocess.run(
            list(scheduler) + [submit_script_name],
            cwd=run_dir,
            capture_output=True,
            text=True,
        )
        try:
            submit_process.check_returncode()
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Failed submitting job '{job_name}':\n"
                f"{submit_process.stderr.strip()}"
            ) from err
        # Write Job ID to a file
        matched_submit_msg = re.match(
            r"Submitted batch job (\d+)", submit_process.stdout
        )
        if matched_submit_msg:
            jobid = matched_submit_msg.group(1)
            (run_dir / "jobid.txt").write_text(jobid)
        else:
            logger.warning(
                f"Unable to parse job ID from output: " + submit_process.stdout
            )
            jobid = None
        logger.info(
            f"Submitted job '{job_name}' ({jobid}). "
            f"Output will be written to: {out_file}"
        )
        return submit_process


def _parse_param(value):
    """Parse an additional command-line parameter for template substitutions

    The following values are supported:

    - Integers or floats
    - List of values: "1,2,3"
    - Exclusive range: "0..3" or "0..<3" (the latter is clearer, but "<" is a
      special character in the shell)
    - Inclusive range: "0...3"
    - Exponentiated values: Single numbers like "2**3" or "10**4", or ranges
      like "10**4...6"

    Note: The syntax for ranges is borrowed from the Swift language:
    https://docs.swift.org/swift-book/documentation/the-swift-programming-language/basicoperators/#Range-Operators
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    # Exponent prefix: 2**x or 10**x, where x is parsed recursively
    match = re.match(r"(\d+)[*]{2}(.+)$", value)
    if match:
        logger.debug(f"'{value}' is exponentiated")
        base = int(match.group(1))
        exponent = _parse_param(match.group(2))
        try:
            # Single exponentiated number
            return base**exponent
        except TypeError:
            # Exponent is a range
            return [base**exponent_i for exponent_i in exponent]
    # List
    value_list = value.strip(",[]").split(",")
    if len(value_list) > 1:
        logger.debug(f"'{value}' is a list")
        return [_parse_param(element.strip()) for element in value_list]
    # Exclusive range: 0..3 or 0..<3 (the latter is clearer, but '<' is a
    # special character in the shell)
    match = re.match(r"(-?\d+)[.]{2}[<]?(-?\d+)$", value)
    if match:
        logger.debug(f"'{value}' is an exclusive range")
        return range(int(match.group(1)), int(match.group(2)))
    # Inclusive range: 0...3
    match = re.match(r"(-?\d+)[.]{3}(-?\d+)$", value)
    if match:
        logger.debug(f"'{value}' is an inclusive range")
        return range(int(match.group(1)), int(match.group(2)) + 1)
    # Integers
    match = re.match(r"(-?\d+)$", value)
    if match:
        logger.debug(f"'{value}' is an int")
        return int(match.group(1))
    # Floats
    match = re.match(r"(-?\d+[.]\d*)$", value)
    if match:
        logger.debug(f"'{value}' is a float")
        return float(match.group(1))
    return value


def _parse_params(ctx, param, all_values):
    if all_values is None:
        return {}
    params = {}
    for value in all_values:
        key_and_value = value.split("=")
        if len(key_and_value) != 2:
            raise click.BadParameter(
                f"The value of '{value}' could not be parsed as a key-value "
                "pair. It should have a single '=' or none."
            )
        params[key_and_value[0]] = _parse_param(key_and_value[1])
    return params


def scheduler_options(f):
    """CLI options for the 'schedule' function.

    These options can be reused by other CLI commands that call the 'schedule'
    function.
    """

    @click.option(
        "--executable",
        "-E",
        show_default="executable listed in input file",
        help=(
            "The executable to run. Can be a path, or just the name of the"
            " executable if it's in the 'PATH'. If unspecified, the"
            " 'Executable' listed in the input file metadata is used."
        ),
    )
    @click.option(
        "--run-dir",
        "-o",
        # No `type=click.Path` because this can be a Jinja template
        help=(
            "The directory to which input file, submit script, etc. are "
            "copied, relative to which the executable will run, and to "
            "which output files are written. "
            "Defaults to the current working directory if the input file is "
            "already there. "
            "Mutually exclusive with '--segments-dir' / '-O'."
        ),
    )
    @click.option(
        "--segments-dir",
        "-O",
        # No `type=click.Path` because this can be a Jinja template
        help=(
            "The directory in which to create the next segment. "
            "Requires '--from-checkpoint' or '--from-last-checkpoint' "
            "unless starting the first segment."
        ),
    )
    @click.option(
        "--create-bin/--no-create-bin",
        default=None,
        help=(
            "Create a bin directory for the simulation and run from it (see "
            "main help text). "
            "(1) When no flag is specified: create one if the run will be "
            "continued and submitted, i.e. if a pipeline directory or "
            "'--segments-dir' / '-O' is set and a scheduler is used. "
            "(2) When '--create-bin' is specified: create one also for a plain "
            "'--run-dir' / '-o' run. "
            "(3) When '--no-create-bin' is specified: run from the build "
            "directory."
        ),
    )
    @click.option(
        "--clean-output",
        "-C",
        is_flag=True,
        help=(
            "Clean up existing output files in the run directory "
            "before running the executable. "
            "See the 'spectre clean-output' command for details."
        ),
    )
    @click.option(
        "--force",
        "-f",
        is_flag=True,
        help=(
            "Overwrite existing files in the '--run-dir' / '-o'. "
            "You may also want to use '--clean-output'."
        ),
    )
    @click.option(
        "--validate/--no-validate",
        default=True,
        help="Validate or skip the validation of the input file.",
    )
    @click.option(
        "--profile-with",
        type=click.Choice(["hpctoolkit"], case_sensitive=True),
        default=None,
        help=(
            "Set to 'hpctoolkit' to enable profiling with HPCToolkit. No other"
            " profilers are currently supported. The executable must be"
            " compiled so that it's compatible with HPCToolkit (see"
            " https://spectre-code.org/profiling.html). This will modify the"
            " submit script to run the executable with 'hpcrun' and postprocess"
            " the profiling data with 'hpcstruct' and 'hpcprof'."
        ),
    )
    # Scheduling options
    @click.option(
        "--scheduler",
        default=("sbatch" if machine else None),
        show_default=(True if machine else "none"),
        help="The scheduler invoked to queue jobs on the machine.",
    )
    @click.option(
        "--no-schedule",
        is_flag=True,
        help="Run the executable directly, without scheduling it.",
    )
    @click.option(
        "--submit-script-template",
        default=None,
        show_default="support/SubmitTemplate.sh next to the bin directory",
        # No `type=click.Path` because this can be a Jinja template
        help=(
            "Path to a submit script (see main help text). It can be a "
            "[Jinja template]("
            "https://jinja.palletsprojects.com/en/3.0.x/templates/) "
            "(see main help text for possible placeholders)."
        ),
    )
    @click.option(
        "--job-name",
        "-J",
        show_default="executable name",
        help=(
            "A short name for the job "
            "(see main help text for possible placeholders)."
        ),
    )
    @click.option(
        "--num-procs",
        "-j",
        "-c",
        type=_parse_param,
        help=(
            "Number of worker threads. Less than a full node will only set as "
            "many Slurm ntasks-per-node as required by machine configuration. "
            "Mutually exclusive with '--num-nodes' / '-N'."
        ),
    )
    @click.option(
        "--num-nodes", "-N", type=_parse_param, help="Number of nodes"
    )
    @click.option("--queue", help="Name of the queue.")
    @click.option(
        "--time-limit",
        "-t",
        help="Wall time limit. Must be compatible with the chosen queue.",
    )
    @click.option(
        "--param",
        "-p",
        "extra_params",
        multiple=True,
        callback=_parse_params,
        help=(
            "Forward an additional parameter to the input file "
            "and submit script templates. "
            "Can be specified multiple times. "
            "Each entry must be a 'key=value' pair, where the key is "
            "the parameter name. The value can be an int, float, "
            "string, a comma-separated list, an inclusive range "
            "like '0...3', an exclusive range like '0..3' or '0..<3', "
            "or an exponentiated value or range like "
            "'2**3' or '10**4...6'. "
            "If a parameter is a list or range, multiple runs are "
            "scheduled recursively. "
            "You can also use the parameter in the 'job_name' and "
            "in the 'run_dir' or 'segment_dir', and when scheduling "
            "ranges of runs you probably should."
        ),
    )
    @click.option(
        "--submit/--no-submit",
        default=None,
        help=(
            "Submit jobs automatically. If neither option is "
            "specified, a prompt will ask for confirmation before "
            "a job is submitted."
        ),
    )
    @click.option(
        "--context-file-name",
        default="SchedulerContext.yaml",
        show_default=True,
        help="Name of the context file that supports resubmissions.",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.command(
    name="schedule", help=schedule.__doc__.replace("**kwargs", "--params")
)
@click.argument(
    "input_file_template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@scheduler_options
@click.option(
    "--from-checkpoint",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help="Restart from this checkpoint.",
)
@click.option(
    "--from-last-checkpoint",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help="Restart from the last checkpoint in this directory.",
)
def schedule_command(
    from_checkpoint,
    from_last_checkpoint,
    **kwargs,
):
    _rich_traceback_guard = True  # Hide traceback until here
    if from_checkpoint and from_last_checkpoint:
        raise click.UsageError(
            "Specify either '--from-checkpoint' or '--from-last-checkpoint', "
            "not both."
        )
    if from_last_checkpoint:
        segments = list_segments(from_last_checkpoint)
        if segments:
            segment = segments[-1]
        else:
            segment = Segment.match(from_last_checkpoint)
        if segment:
            all_checkpoints = segment.checkpoints
            assert all_checkpoints, (
                f"The segment '{segment}' contains no checkpoints. It may"
                " be incomplete. Did you forget to remove it?"
            )
        else:
            all_checkpoints = list_checkpoints(from_last_checkpoint)
            assert all_checkpoints, (
                f"Directory '{from_last_checkpoint}' contains no checkpoints "
                f"that match the pattern '{Checkpoint.NAME_PATTERN.pattern}'."
            )
        from_checkpoint = all_checkpoints[-1]
    schedule(from_checkpoint=from_checkpoint, **kwargs)


if __name__ == "__main__":
    schedule_command(help_option_names=["-h", "--help"])
