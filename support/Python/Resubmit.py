# Distributed under the MIT License.
# See LICENSE.txt for details.

import subprocess
from pathlib import Path
from typing import Optional

import click
import yaml

from spectre.support.DirectoryStructure import Segment, list_segments
from spectre.support.Schedule import schedule


def resubmit(
    segments_dir: Path, context_file_name: str = "SubmitContext.yaml", **kwargs
) -> Optional[subprocess.CompletedProcess]:
    """Create the next segment in the SEGMENTS_DIR and schedule it

    \f
    Arguments:
      segments_dir: Path to the segments directory, or path to the last segment
        in the segments directory. The next segment will be created here.
      context_file_name: Optional. Name of the file that stores the context
        for resubmissions in the 'run_dir'. This file gets created by
        'spectre.support.schedule'. (Default: "SubmitContext.yaml")

    Returns: The 'subprocess.CompletedProcess' representing the process
      that scheduled the run. Returns 'None' if no run was scheduled.
    """
    # Resolve segments dir (if invoked from within a segment)
    segment = Segment.match(segments_dir)
    if segment:
        segments_dir = segment.path.resolve().parent
    # Resolve last segment
    all_segments = list_segments(segments_dir)
    assert all_segments, (
        f"Directory '{segments_dir}' contains no segments "
        f"that match the pattern '{Segment.NAME_PATTERN.pattern}'."
    )
    last_segment = all_segments[-1]
    if segment:
        assert segment.path.resolve() == last_segment.path.resolve(), (
            f"The specified segment ({segment.path}) is not the last in the"
            f" directory ({last_segment.path}). If you want to resubmit from"
            " the specified segment, remove the later segments and try again."
            " Otherwise, resubmit from the last segment or the parent"
            " (segments) directory."
        )
    context_file = last_segment.path / context_file_name
    # Load context
    with open(context_file, "r") as open_context_file:
        context = yaml.safe_load(open_context_file)
    # Continue from the last checkpoint
    last_segment_checkpoints = last_segment.checkpoints
    assert last_segment_checkpoints, (
        f"The last segment '{last_segment.path}' contains no checkpoints. It"
        " may be incomplete. Did you forget to remove it?"
    )
    context["from_checkpoint"] = last_segment_checkpoints[-1]
    # We resubmit the rendered input file from the last segment instead of
    # trying to render the template again (the input file template wasn't copied
    # to the segments directory, so there's no guarantee it still exists)
    context["input_file_template"] = last_segment.path / context["input_file"]
    # Remove the old `run_dir`. It will be set to the new segment.
    del context["run_dir"]
    # Override the `segments_dir` just to be safe
    context["segments_dir"] = segments_dir
    # Override with CLI options
    context.update(kwargs)
    schedule(**context)


@click.command(help=resubmit.__doc__)
@click.argument(
    "segments_dirs",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    nargs=-1,
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
    default="SubmitContext.yaml",
    show_default=True,
    help="Name of the context file that supports resubmissions.",
)
def resubmit_command(segments_dirs, **kwargs):
    _rich_traceback_guard = True  # Hide tracebacks until here
    for segment_dir in segments_dirs:
        resubmit(segment_dir, **kwargs)


if __name__ == "__main__":
    resubmit_command(help_option_names=["-h", "--help"])
