# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import rich
import rich.syntax
import yaml

logger = logging.getLogger(__name__)


class InvalidInputFileError(Exception):
    """Indicates that the input file cannot be parsed by the executable

    The exception prints the 'message' by default. To print additional context,
    catch the exception and print the 'render_context()' as well.

    Attributes:
      input_file_path: Path to the input file on disk.
      line_number: Line number in the input file where the parse error occurred.
      yaml_path: Sequence of YAML keys in the input file that lead to the
        parse error, such as '["DomainCreator", "Interval", "LowerBound"]'.
      message: Error message emitted by the option parsing.
    """

    def __init__(
        self,
        input_file_path: Union[str, Path],
        line_number: Optional[int],
        yaml_path: Sequence[str],
        message: Optional[str],
    ):
        self.input_file_path = Path(input_file_path)
        self.line_number = line_number
        self.yaml_path = yaml_path
        self.message = message
        super().__init__(message)

    @rich.console.group()
    def render_context(self) -> rich.console.RenderResult:
        if self.line_number is not None:
            yield f"{self.input_file_path.resolve()}:{self.line_number}"
            yield rich.syntax.Syntax(
                self.input_file_path.read_text(),
                theme="ansi_dark",
                lexer="yaml",
                line_range=(self.line_number - 2, self.line_number + 2),
                highlight_lines={self.line_number},
                line_numbers=True,
            )
            yield ""
        if self.yaml_path:
            yield "In [bold]" + ".".join(self.yaml_path) + ":"
            yield ""


def validate_input_file(
    input_file_path: Union[str, Path],
    executable: Optional[Union[str, Path]] = None,
    work_dir: Optional[Union[str, Path]] = None,
    print_context: bool = True,
    raise_exception: bool = True,
):
    """Check an input file for parse errors

    Invokes the executable with the '--check-options' flag to check for parse
    errors.

    Arguments:
      input_file_path: Path to the input file on disk.
      executable: Name or path of the executable. If unspecified, use the
        'Executable:' in the input file metadata.
      work_dir: Working directory for invoking the executable with the input
        file. Relative paths in the input file are resolved from here.
      print_context: Print additional context where the parse error occurred
        before raising an exception (default: True).
      raise_exception: Raise an 'InvalidInputFileError' if a parse error
        occurred (default: True).
    """
    input_file_path = Path(input_file_path)

    # Resolve the executable
    if not executable:
        with open(input_file_path, "r") as open_input_file:
            executable = next(yaml.safe_load_all(open_input_file))["Executable"]

    # Use the executable to validate the input file
    process = subprocess.run(
        [executable, "--input-file", input_file_path, "--check-options"],
        capture_output=True,
        text=True,
        cwd=work_dir,
    )

    if process.returncode == 0:
        return

    # Parse the validation error
    logger.debug(process.stderr)
    found_hints = False
    path = []
    line_number = None
    col = None
    msg = []
    for line in process.stderr.split("\n"):
        if str(input_file_path) in line:
            found_hints = True
            continue
        if found_hints:
            if line.startswith("In group"):
                path.append(line[9:-1])
            elif line.startswith("While parsing option"):
                path.append(line[21:-1])
            elif line.startswith("While creating a"):
                path.append(line[17:-1])
            elif line.startswith("At line"):
                match = re.match(r"At line ([0-9]+) column ([0-9]+)", line)
                line_number, _ = map(int, match.groups())
            elif line.startswith("While operating factory for"):
                # remove "unique_ptr" entry from path
                path.pop()
            elif "ERROR" in line:
                break
            else:
                msg.append(line)

    # Print context and raise exception
    error = InvalidInputFileError(
        input_file_path=input_file_path,
        line_number=line_number,
        yaml_path=path,
        message="\n".join(msg).strip(),
    )
    if print_context:
        rich.print(error.render_context())
    if raise_exception:
        _rich_traceback_guard = True
        raise error


@click.command()
@click.argument(
    "input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--executable",
    "-e",
    help=(
        "Name or path of the executable. "
        "If unspecified, the 'Executable:' in the input file "
        "metadata is used."
    ),
)
def validate_input_file_command(**kwargs):
    """Check an input file for parse errors"""
    validate_input_file(**kwargs)


if __name__ == "__main__":
    validate_input_file_command(help_option_names=["-h", "--help"])
