# Distributed under the MIT License.
# See LICENSE.txt for details.

import datetime
import glob
import logging
import os
import re
import subprocess
import time
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Sequence

import click
import humanize
import pandas as pd
import rich.console
import rich.live
import rich.rule
import rich.table
import yaml

from spectre.support.DirectoryStructure import Segment

from .ExecutableStatus import match_executable_status

logger = logging.getLogger(__name__)


def fetch_job_data(
    fields: Sequence[str],
    user: Optional[str],
    allusers: bool = False,
    state: Optional[str] = None,
    starttime: Optional[str] = None,
) -> pd.DataFrame:
    """Query Slurm 'sacct' to get metadata of recent jobs on the machine.

    Arguments:
      fields: List of Slurm fields that 'sacct --format' accepts.
        Run 'sacct --helpformat' to print all available fields.
      user: Slurm user IDs or names, or None for the current user.
        See documentation for 'sacct -u' for details.
      allusers: Fetch data for all users.
        See documentation for 'sacct -a' for details.
      state: Fetch only jobs with this state.
        See documentation for 'sacct -s' for details.
      starttime: Fetch only jobs after this time.
        See documentation for 'sacct -S' for details.

    Returns: Pandas DataFrame with the job data.
    """
    completed_process = subprocess.run(
        ["sacct", "-PX", "--format", ",".join(fields)]
        + (["-u", user] if user else [])
        + (["-a"] if allusers else [])
        + (["-s", state] if state else [])
        + (["-S", starttime] if starttime else []),
        capture_output=True,
        text=True,
    )
    try:
        completed_process.check_returncode()
    except subprocess.CalledProcessError as err:
        raise ValueError(completed_process.stderr) from err
    job_data = pd.read_table(
        StringIO(completed_process.stdout), sep="|", keep_default_na=False
    )
    # Parse dates and times. Do this in postprocessing because
    # `pd.read_table(parse_dates=...)` doesn't handle NaN values well.
    date_cols = set(fields).intersection({"Start", "End"})
    for date_col in date_cols:
        job_data[date_col] = job_data[date_col].apply(
            lambda v: v.replace("Unknown", "NaN")
        )
        # infer_datetime_format is deprecated in version 2.0.0
        # so just use the actual SLURM format
        job_data[date_col] = pd.to_datetime(
            job_data[date_col], format="%Y-%m-%dT%H:%M:%S"
        )
    # We could parse the elapsed time as a timedelta, but the string
    # representation is fine so we don't right now. Here's the code for it:
    # if "Elapsed" in fields:
    #     job_data["Elapsed"] = pd.to_timedelta(
    #         job_data["Elapsed"].apply(lambda v: v.replace("-", " days ")))
    return job_data


def get_input_file(comment: Optional[str], work_dir: str) -> Optional[str]:
    """Find the input file corresponding to a job.

    Arguments:
      comment: The Slurm comment field. The input file is extracted from it if
        it includes a line like "SPECTRE_INPUT_FILE=path/to/input/file".
      work_dir: The working directory of the job. If no input file was found
        in the Slurm comment, we see if there's a single YAML file in the
        work dir and assume that's the input file.

    Returns: Path to the input file, or None.
    """
    if comment:
        # Get the input file from the Slurm comment if the submission
        # specified it
        match = re.search(
            r"^SPECTRE_INPUT_FILE=(.*)", comment, flags=re.MULTILINE
        )
        if match:
            return os.path.join(work_dir, match.group(1))
        else:
            logger.debug(
                "Could not find 'SPECTRE_INPUT_FILE' in Slurm comment:\n"
                + comment
            )
    # Fallback: Check if there's a single YAML file in the work dir
    yaml_files = glob.glob(os.path.join(work_dir, "*.yaml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    else:
        logger.debug(
            "No input file found. "
            f"Didn't find a single YAML file in '{work_dir}'. "
            f"YAML files found: {yaml_files}"
        )
        return None


def get_executable_name(
    comment: Optional[str], input_file_path: Optional[str]
) -> Optional[str]:
    """Determine the executable name of a job.

    Arguments:
      comment: The Slurm comment field. The executable name is extracted from it
        if it includes a line like "SPECTRE_EXECUTABLE=path/to/executable".
      input_file_path: Path to input file. If no executable name was found in
        the Slurm comment, we try to extract it from the input file.

    Returns: Executable name, or None.
    """
    if comment:
        # Get the executable from the Slurm comment if the submission
        # specified it
        match = re.search(
            r"^SPECTRE_EXECUTABLE=(.*)", comment, flags=re.MULTILINE
        )
        if match:
            return os.path.basename(match.group(1))
        else:
            logger.debug(
                "Could not find 'SPECTRE_EXECUTABLE' in Slurm comment:\n"
                + comment
            )
    # Fallback: See if the executable is specified in the input file metadata
    if not input_file_path:
        return None
    with open(input_file_path, "r") as open_input_file:
        metadata = next(yaml.safe_load_all(open_input_file))
    if metadata and "Executable" in metadata:
        return os.path.basename(metadata["Executable"])
    # Backwards compatibility for input files without metadata (can be removed
    # as soon as most people have rebased)
    match = re.search(
        r"#\s+Executable:\s+(.+)", Path(input_file_path).read_text()
    )
    if match:
        return match.group(1)
    return None


def _state_order(state):
    order = [
        "RUNNING",
        "PENDING",
        "COMPLETED",
        "TIMEOUT",
        "FAILED",
        "CANCELLED",
    ]
    try:
        return order.index(state)
    except ValueError:
        return None


def _format(field: str, value: Any, state_styles: dict) -> str:
    if field == "State":
        style = {
            "RUNNING": "[blue]",
            "COMPLETED": "[green]",
            "PENDING": "[magenta]",
            "FAILED": "[red]",
            "TIMEOUT": "[red]",
            "CANCELLED": "[red]",
        }
        style.update(state_styles)
        return style.get(value, "") + str(value)
    elif field in ["Start", "End"]:
        if pd.isnull(value):
            return "-"
        else:
            return humanize.naturaldate(value) + " " + value.strftime("%X")
    else:
        return str(value)


@rich.console.group()
def render_status(
    show_paths,
    show_unidentified,
    show_deleted,
    show_all_segments,
    state_styles,
    **kwargs,
):
    job_data = fetch_job_data(
        [
            "JobID",
            "User",
            "JobName",
            "NCPUS",
            "NNodes",
            "Elapsed",
            "End",
            "State",
            "WorkDir",
            "Comment",
        ],
        **kwargs,
    )

    # Do nothing if job list is empty
    if len(job_data) == 0:
        return

    # Remove deleted jobs
    if not show_deleted:
        deleted_jobs = job_data[~job_data["WorkDir"].map(os.path.exists)]
        job_data.drop(deleted_jobs.index, inplace=True)

    # Keep only latest in a series of segments
    job_data[["SegmentsDir", "SegmentId"]] = [
        (
            (str(segment.path.resolve().parent), segment.id)
            if segment
            else (None, None)
        )
        for segment in job_data["WorkDir"].map(Segment.match)
    ]
    if not show_all_segments:
        for segments_dir in pd.unique(job_data["SegmentsDir"]):
            if not segments_dir:
                continue
            segments_jobs = job_data[job_data["SegmentsDir"] == segments_dir]
            if len(segments_jobs) == 1:
                continue
            latest_segment_id = sorted(segments_jobs["SegmentId"])[-1]
            drop_segment_jobs = job_data[
                (job_data["SegmentsDir"] == segments_dir)
                & (job_data["SegmentId"] != latest_segment_id)
            ]
            job_data.drop(drop_segment_jobs.index, inplace=True)

    # List most recent jobs first
    job_data.sort_values("JobID", inplace=True, ascending=False)

    # Get the input file corresponding to each job
    job_data["InputFile"] = [
        get_input_file(comment, work_dir)
        for comment, work_dir in zip(job_data["Comment"], job_data["WorkDir"])
    ]

    # Get the executable name corresponding to each job.
    job_data["ExecutableName"] = [
        get_executable_name(comment, input_file)
        for comment, input_file in zip(
            job_data["Comment"], job_data["InputFile"]
        )
    ]

    # Invalidate older jobs that ran in the same work dir because they would
    # report wrong information (that of the newer job).
    for work_dir in pd.unique(job_data["WorkDir"]):
        duplicate_jobs = job_data[job_data["WorkDir"] == work_dir]
        if len(duplicate_jobs) == 1:
            continue
        # Jobs are already sorted by JobID, so just keep the latest job
        # associated and invalidate the others.
        latest_job_id = duplicate_jobs.iloc[0]["JobID"]
        job_data.loc[
            (job_data["WorkDir"] == work_dir)
            & (job_data["JobID"] != latest_job_id),
            ["WorkDir", "NewJobID"],
        ] = [None, latest_job_id]

    # Normalize "cancelled" job state (remove "by X")
    job_data.loc[job_data["State"].str.contains("CANCELLED"), "State"] = (
        "CANCELLED"
    )

    # Add metadata so jobs can be grouped by state
    job_data["StateOrder"] = job_data["State"].apply(_state_order)

    # We'll print these columns
    standard_fields = [
        "State",
        "End",
        "JobID",
        "JobName",
        "Elapsed",
        "NCPUS",
        "NNodes",
    ]
    if kwargs["allusers"]:
        standard_fields.insert(2, "User")
    # Transform some column names for better readability
    col_names = {
        "NCPUS": "Cores",
        "NNodes": "Nodes",
    }
    standard_columns = [col_names.get(col, col) for col in standard_fields]

    # Group output by executable
    first_section = True
    for executable_name, exec_data in job_data.groupby("ExecutableName"):
        if first_section:
            first_section = False
        else:
            yield ""
        yield rich.rule.Rule(f"[bold]{executable_name}", align="left")
        yield ""
        executable_status = match_executable_status(executable_name)

        extra_columns = [
            (field + f" [{unit}]") if unit else field
            for field, unit in executable_status.fields.items()
        ]
        columns = standard_columns + extra_columns
        table = rich.table.Table(*columns, box=None)

        # Group by job state
        for state_index, data in exec_data.groupby("StateOrder"):
            for _, row in data.iterrows():
                # Extract job status and format row for output
                row_formatted = [
                    _format(field, row[field], state_styles)
                    for field in standard_fields
                ]
                try:
                    with open(row["InputFile"], "r") as open_input_file:
                        # Backwards compatibility for input files without
                        # metadata. Once people have rebased this can be changed
                        # to: _, input_file = yaml.safe_load_all(...)
                        for doc in yaml.safe_load_all(open_input_file):
                            input_file = doc
                except:
                    logger.debug("Unable to load input file.", exc_info=True)
                    input_file = None
                try:
                    status = executable_status.status(
                        input_file, row["WorkDir"]
                    )
                except:
                    logger.debug(
                        "Unable to extract executable status.", exc_info=True
                    )
                    status = {}
                row_formatted += [
                    (
                        executable_status.format(field, status[field])
                        if field in status
                        else "-"
                    )
                    for field in executable_status.fields.keys()
                ]
                table.add_row(*row_formatted)

                # Print paths if requested
                if show_paths:
                    yield table
                    # Print WorkDir in its own line so it wraps nicely in the
                    # terminal and can be copied
                    yield (
                        " [bold]WorkDir:[/bold] "
                        + (
                            row["WorkDir"]
                            if row["WorkDir"]
                            else (
                                "[italic]Same as job"
                                f" {row['NewJobID']}[/italic]"
                            )
                        )
                    )
                    yield " [bold]InputFile:[/bold] " + str(row["InputFile"])
                    table = rich.table.Table(*columns, box=None)
        if not show_paths:
            yield table

    # Output jobs that couldn't be parsed
    unidentified_jobs = job_data[job_data["ExecutableName"].isnull()]
    if len(unidentified_jobs) > 0 and show_unidentified:
        yield ""
        yield rich.rule.Rule("[bold]Unidentified Jobs", align="left")
        yield ""
        table = rich.table.Table(*standard_columns, box=None)
        for i, row in unidentified_jobs.iterrows():
            row_formatted = [
                _format(field, row[field], state_styles)
                for field in standard_fields
            ]
            table.add_row(*row_formatted)
        yield table


@click.command()
@click.option(
    "-u",
    "--uid",
    "--user",
    "user",
    show_default="you",
    help="User name or user ID. See documentation for 'sacct -u' for details.",
)
@click.option(
    "-a",
    "--allusers",
    is_flag=True,
    help=(
        "Show jobs for all users. See documentation for 'sacct -a' for details."
    ),
)
@click.option(
    "-p",
    "--show-paths",
    is_flag=True,
    help="Show job working directory and input file paths.",
)
@click.option(
    "-U",
    "--show-unidentified",
    is_flag=True,
    help="Also show jobs that were not identified as SpECTRE executables.",
)
@click.option(
    "-D",
    "--show-deleted",
    is_flag=True,
    help="Also show jobs that ran in directories that are now deleted.",
)
@click.option(
    "-A",
    "--show-all-segments",
    is_flag=True,
    help="Show all segments instead of just the latest.",
)
@click.option(
    "-s",
    "--state",
    help=(
        "Show only jobs with this state, "
        "e.g. running (r) or completed (cd). "
        "See documentation for 'sacct -s' for details."
    ),
)
@click.option(
    "-S",
    "--starttime",
    show_default="start of today",
    help=(
        "Show jobs eligible after this time, e.g. 'now-2days'. "
        "See documentation for 'sacct -S' for details."
    ),
)
@click.option(
    "-w",
    "--watch",
    "refresh_rate",
    type=float,
    default=None,
    help=(
        "On a new screen, refresh jobs every 'watch' number "
        "of seconds. Exit out with Ctl+C."
    ),
)
@click.option(
    "--state-styles",
    type=dict,
    default={},
    help=(
        "Dictionary between sacct states and rich modifiers for "
        "how the state will be printed. Rather than always having "
        "to specify a dict on the command line, you can add this "
        "to the spectre config file. \n\n"
        "An example for the config file would be\n\n"
        "\b\n"
        "status:\n"
        "  state_styles:\n"
        "    RUNNING: '[green]'\n"
        "    COMPLETED: '[bold][red]'\n\n"
        "See `spectre -h` for its path."
    ),
)
def status_command(refresh_rate, **kwargs):
    """Gives an overview of simulations running on this machine."""

    # Start printing things
    console = rich.console.Console()

    if not refresh_rate:
        console.print(render_status(**kwargs))
        return

    try:
        logging.disable()
        with rich.live.Live(
            render_status(**kwargs),
            console=console,
            auto_refresh=False,
            screen=True,
        ) as live:
            while True:
                time.sleep(refresh_rate)
                live.update(render_status(**kwargs), refresh=True)
    except KeyboardInterrupt:
        pass
