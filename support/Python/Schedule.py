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
import jinja2.meta
import numpy as np
import yaml
from rich.pretty import pretty_repr
from yaml.representer import SafeRepresenter

from spectre.support.DirectoryStructure import (
    Checkpoint,
    Segment,
    list_checkpoints,
    list_segments,
)
from spectre.support.Machines import this_machine
from spectre.support.RunNext import run_next
from spectre.tools.ValidateInputFile import validate_input_file
from spectre.Visualization.ReadInputFile import find_phase_change

logger = logging.getLogger(__name__)

# CMake configures the submit script templates for the current machine to this
# path
default_submit_script_template = Path(__file__).parent / "SubmitTemplate.sh"


def _resolve_executable(executable: Union[str, Path]) -> Path:
    """Look up the 'executable' in the PATH

    Raises 'ValueError' if the executable is not found.
    """
    logger.debug(f"Resolving executable: {executable}")
    # This default bin dir is already added in spectre.__main__.py, but only
    # when running the CLI. It is the bin dir of the build directory that
    # contains this script. When running Python code outside the CLI this should
    # also be the default bin dir.
    default_bin_dir = Path(__file__).parent.parent.parent.parent.resolve()
    path = os.environ["PATH"] + ":" + str(default_bin_dir)
    which_exec = shutil.which(executable, path=path)
    if not which_exec:
        raise ValueError(
            f"Executable not found: {executable}. Make sure it is compiled. To"
            " look for executables in a specific build directory make sure it"
            " is in the 'PATH' or use the 'spectre --build-dir / -b' option."
        )
    return Path(which_exec).resolve()


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


def _copy_to_dir(src_file: Path, dest_dir: Path, force: bool = False) -> Path:
    """Copy the 'src_file' to the 'dest_dir', keeping the file name the same

    Returns the path to the new file.
    """
    assert src_file.is_file()
    assert dest_dir.is_dir()
    if src_file.resolve().parent == dest_dir.resolve():
        return src_file
    dest = (dest_dir / src_file.name).resolve()
    if dest.exists() and not force:
        raise OSError(
            f"File already exists at '{dest}'. Retry with "
            "'force' ('--force' / '-f') to overwrite."
        )
    logging.debug(f"Copy file: {src_file} -> {dest}")
    shutil.copy(src_file, dest)
    return dest


def _copy_submit_script_template(
    submit_script_template: Path,
    dest_dir: Path,
    template_env: jinja2.Environment,
    force: bool = False,
) -> Path:
    dest = _copy_to_dir(submit_script_template, dest_dir, force=force)
    # Also copy all referenced templates (the "SubmitBase.sh" parent template)
    syntax_tree = template_env.parse(dest.read_text())
    for referenced_template in jinja2.meta.find_referenced_templates(
        syntax_tree
    ):
        referenced_template_src = (
            submit_script_template.resolve().parent / referenced_template
        )
        _copy_to_dir(referenced_template_src, dest_dir, force=force)
    return dest


# Write `pathlib.Path` objects to YAML as plain strings
def _path_representer(dumper: yaml.Dumper, path: Path) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(path))


# Write `numpy.float64` as regular floats
def _numpy_representer(
    dumper: yaml.Dumper, value: np.float64
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(value))


def schedule(
    input_file_template: Union[str, Path],
    scheduler: Optional[Union[str, Sequence]],
    no_schedule: Optional[bool] = None,
    executable: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    copy_executable: Optional[bool] = None,
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
    no_validate=False,
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

    By default, the executable and submit scripts will be copied to the segments
    directory to support resubmissions (see below). See the 'copy_executable'
    argument docs for details on controlling this behavior.

    # Segments and run directories

    You can set either the 'run_dir' or the 'segments_dir' to specify where the
    executable will run (but not both). If you specify a 'run_dir', the
    executable will run in it directly. If you specify a 'segments_dir', a new
    segment will be created and used as the 'run_dir'. Segments are named with
    incrementing integers and continue the run from the previous segment. For
    example, the following is a typical 'segments_dir':

    \b
    ```sh
    # Copy of the executable
    MyExecutable
    # Copy of the submit script template (base and machine-specific)
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
          'executable' path).

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
      copy_executable: Copy the executable to the run or segments directory.
        By default (when set to 'None'):
          - If '--run-dir' / '-o' is set, don't copy.
          - If '--segments-dir' / '-O' is set, copy to segments directory to
            support resubmission.
        When set to 'True':
          - If '--run-dir' / '-o' is set, copy to the run directory.
          - If '--segments-dir' / '-O' is set, copy to segments directory to
            support resubmission. Still don't copy to individual segments.
        When set to 'False': Never copy.
      job_name: Optional. A string describing the job.
        Can be a Jinja template (see above). (Default: executable name)
      submit_script_template: Optional. Path to a submit script. It will be
        copied to the 'run_dir' if a 'scheduler' is set. Can be a Jinja template
        (see above). (Default: value of 'default_submit_script_template')
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
    if scheduler and not submit_script_template:
        submit_script_template = default_submit_script_template
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

    # Set up template environment with basic configuration
    template_env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
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
    executable = _resolve_executable(executable)
    logger.info(f"Running with executable: {executable}")
    # Only set executable name because the path may change if we copy it later
    context.update(executable_name=executable.name)

    # Resolve number of cores, nodes, etc.
    num_procs = kwargs.get("num_procs")
    num_nodes = kwargs.get("num_nodes")
    if num_procs:
        assert (
            num_nodes is None or num_nodes == 1
        ), "Specify either 'num_procs' or 'num_nodes', not both."
        num_nodes = 1
    # Set the context variables only if defined, so they don't print as "None"
    # https://jinja.palletsprojects.com/en/3.0.x/templates/#jinja-filters.default
    if num_procs:
        context.update(num_procs=num_procs)
    if num_nodes:
        context.update(num_nodes=num_nodes)

    # Resolve job_name
    if job_name:
        job_name = template_env.from_string(job_name).render(context).strip()
    else:
        job_name = executable.name
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

    if not no_validate:
        # Validate input file
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

    # Copy executable to run directory if enabled
    if copy_executable and not segments_dir:
        executable = _copy_to_dir(executable, run_dir, force=force)

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
        run_command = (machine.launch_command if machine else []) + [
            str(executable),
            "--input-file",
            str(input_file_path.resolve()),
        ]
        if auto_provision:
            run_command += ["+auto-provision"]
        else:
            run_command += ["+p", str(num_procs)]
        if from_checkpoint:
            run_command += ["+restart", str(from_checkpoint)]
        logger.debug(f"Run command: {run_command}")
        if submit is False:
            return
        process = subprocess.Popen(run_command, cwd=run_dir)
        # Realtime streaming of _captured_ stdout and stderr to the console
        # doesn't seem to work reliably, so we just let the process stream
        # directly to the console and wait for it to complete.
        process.wait()
        # Raise errors on non-zero exit codes
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=run_command
            )
        # Run the 'Next' entrypoint listed in the input file metadata
        if metadata and "Next" in metadata:
            run_next(
                metadata["Next"], input_file_path=input_file_path, cwd=run_dir
            )
        return process

    # Copy executable to segments directory
    if (copy_executable or copy_executable is None) and segments_dir:
        executable = _copy_to_dir(executable, segments_dir, force=force)
    context.update(executable=executable, copy_executable=copy_executable)

    # Resolve CLI for resubmissions
    # This is the path of the `spectre` CLI where this script is installed.
    # We may have to make this more robust for resubmissions if we run into
    # problems or unexpected behavior.
    spectre_cli = Path(__file__).parent.parent.parent.parent / "spectre"
    if spectre_cli:
        context.update(spectre_cli=spectre_cli)

    # Configure submit script
    submit_script_template = Path(submit_script_template).resolve()
    if segments_dir:
        submit_script_template = _copy_submit_script_template(
            submit_script_template,
            segments_dir,
            template_env=template_env,
            force=force,
        )
    context.update(submit_script_template=submit_script_template)
    logger.debug(
        f"Configure submit script template '{submit_script_template}' with"
        f" these parameters: {pretty_repr(context)}"
    )
    # Use a FileSystemLoader to support template inheritance
    submit_script_template_env = template_env.overlay(
        loader=jinja2.FileSystemLoader(submit_script_template.parent)
    )
    rendered_submit_script = submit_script_template_env.get_template(
        submit_script_template.name
    ).render(context)
    submit_script_path = run_dir / submit_script_name
    _write_or_overwrite(rendered_submit_script, submit_script_path, force=force)

    # Write context to file to support resubmissions
    if segments_dir:
        with open(run_dir / context_file_name, "w") as open_context_file:
            yaml_dumper = yaml.SafeDumper
            yaml_dumper.add_multi_representer(Path, _path_representer)
            yaml_dumper.add_multi_representer(np.float64, _numpy_representer)
            yaml.dump(context, open_context_file, Dumper=yaml_dumper)

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
        "--copy-executable/--no-copy-executable",
        default=None,
        help=(
            "Copy the executable to the run or segments directory. "
            "(1) When no flag is specified: "
            "If '--run-dir' / '-o' is set, don't copy. "
            "If '--segments-dir' / '-O' is set, copy to segments "
            "directory to support resubmission. "
            "(2) When '--copy-executable' is specified: "
            "If '--run-dir' / '-o' is set, copy to the run "
            "directory. "
            "If '--segments-dir' / '-O' is set, copy to segments "
            "directory to support resubmission. Still don't copy to "
            "individual segments. "
            "(3) When '--no-copy-executable' is specified: "
            "Never copy."
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
        "--no-validate",
        is_flag=True,
        help="Skip validation of the input file.",
    )
    # Scheduling options
    @click.option(
        "--scheduler",
        default=("sbatch" if default_submit_script_template.exists() else None),
        show_default=(
            True if default_submit_script_template.exists() else "none"
        ),
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
        show_default=str(default_submit_script_template),
        # No `type=click.Path` because this can be a Jinja template
        help=(
            "Path to a submit script. "
            "It will be copied to the 'run_dir'. It can be a [Jinja template]("
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
            "Number of worker threads. "
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
