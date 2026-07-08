# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Expose the 'spectre' CLI as a Model Context Protocol (MCP) server.

The 'spectre mcp' subcommand starts a stdio MCP server that auto-generates one
MCP tool per leaf command of the 'spectre' CLI. The tools are discovered by
walking the existing 'click' command tree, so the server stays in sync with the
CLI automatically. An MCP client (such as Claude Code) discovers the tools at
runtime via the protocol and invokes them; each tool call runs the corresponding
'spectre' command and returns its output.

Why a custom implementation instead of an off-the-shelf library
---------------------------------------------------------------
Several libraries (e.g. 'click-mcp') turn a 'click' CLI into an MCP server with
a single decorator. We deliberately do not use one, because the convenience
comes at the cost of control over exactly what an agent can reach and how
commands run. The tradeoff is that we maintain a bit more code and keep it in
sync with the CLI ourselves; in exchange we get:

- Curated, safe-by-default exposure (the main reason). A one-line decorator
  exposes *every* command as a tool, with no allowlist. Many 'spectre' commands
  submit cluster jobs, drive auto-resubmitting multi-stage pipelines, or
  irreversibly delete data -- actions an agent should not trigger from a single
  tool call. 'discover_commands' hides those by default ('DEFAULT_UNSAFE',
  selected via 'expose="safe"'), with explicit opt-in through '--expose all',
  '--include', and '--exclude'. We decide what the agent can reach rather than
  accepting the CLI's full surface area.

- Subprocess isolation that protects the protocol channel. stdout is reserved
  for the MCP protocol, but commands that spawn executables write to the OS file
  descriptors they inherit and would corrupt it. The default execution path
  ('run_tool_subprocess') runs each command in a subprocess and captures all
  output -- including any child process output -- in a pipe. ('--in-process' is
  available as a faster path for pure-Python commands.) Libraries that capture
  stdout in-process do not guard against child-process output.

- Bounded output for the client's context window. 'MAX_OUTPUT_CHARS' head/tail
  truncation keeps large dumps (e.g. volume data) from blowing up the client.

- No hard dependency and a testable core (see below).

- 'click' as the single source of truth for validation. We generate JSON
  schemas that preserve choices, int/float ranges, and 'multiple'/'nargs'
  arity, but run the server with 'validate_input=False' so 'click' re-validates
  every argument and produces the better error messages.

The helper functions in this module (command discovery, JSON-schema generation,
argument reconstruction, and command execution) deliberately do not import the
'mcp' package at module load. Only 'mcp_command' imports it, lazily, so the rest
of the module can be imported and unit-tested without the optional dependency.
"""

import functools
import json
import logging
import sys
from collections import namedtuple

import click

logger = logging.getLogger(__name__)

# Cap on the number of characters of a single stream (stdout or stderr) returned
# in one tool result. Commands can produce very large dumps (e.g. volume data),
# and an unbounded result would blow up the MCP client's context window. When a
# stream exceeds the cap we keep its head and tail and replace the middle with a
# marker, so both the start of the output and any trailing error stay visible.
MAX_OUTPUT_CHARS = 100_000

# Commands excluded from the default ("safe") exposure. These submit cluster
# jobs, drive auto-resubmitting multi-stage pipelines, or irreversibly delete
# data, so we don't want an MCP client to trigger them with a single tool call
# by default. They are re-enabled with '--expose all' or by listing them in
# '--include'. Names are the generated tool names (see 'tool_name').
DEFAULT_UNSAFE = frozenset(
    {
        "resubmit",
        "run_next",
        "schedule",
        "bbh_eccentricity_control",
        "bbh_generate_id",
        "bbh_postprocess_id",
        "bbh_start_inspiral",
        "bbh_start_ringdown",
        # Destructive or arbitrary-code operations: rewrite/delete data in
        # place or execute user-named kernels.
        "clean_output",
        "delete_subfiles",
        "transform_volume_data",
    }
)

# A leaf 'click' command exposed as an MCP tool.
# - tool_name: the MCP tool name, e.g. "bbh_find_horizon".
# - path: the CLI command path, e.g. ["bbh", "find-horizon"].
# - command: the resolved 'click.Command'.
DiscoveredCommand = namedtuple(
    "DiscoveredCommand", ["tool_name", "path", "command"]
)


def _normalize(name: str) -> str:
    """Map a command name to its tool-name spelling (dashes to underscores)."""
    return name.replace("-", "_")


def tool_name(path) -> str:
    """Build the MCP tool name for a nested CLI command path.

    For example, ['bbh', 'generate-id'] becomes 'bbh_generate_id'.
    """
    return "_".join(_normalize(segment) for segment in path)


def discover_commands(root_command, *, expose="safe", include=(), exclude=()):
    """Walk the 'click' command tree and return the commands to expose as tools.

    Groups are recursed into; leaf commands become tools. Subcommands are
    resolved lazily (which may trigger module imports); a command that fails to
    import (e.g. an optional 'paraview' dependency is missing) is skipped with a
    warning rather than aborting the whole server.

    Arguments:
      root_command: The root 'click.Group' to walk (the 'spectre' CLI).
      expose: "safe" to hide the 'DEFAULT_UNSAFE' commands, or "all".
      include: Command names to expose even in "safe" mode.
      exclude: Command names to always hide.

    Returns a list of 'DiscoveredCommand', sorted by tool name.
    """
    include = {_normalize(name) for name in (include or ())}
    exclude = {_normalize(name) for name in (exclude or ())}
    discovered = []

    def _walk(command, path):
        info_name = path[-1] if path else command.name
        ctx = click.Context(command, info_name=info_name)
        try:
            names = command.list_commands(ctx)
        except Exception as error:
            logger.warning(
                "Could not list subcommands of '%s': %s",
                "/".join(path) or command.name,
                error,
            )
            return
        for name in names:
            try:
                subcommand = command.get_command(ctx, name)
            except Exception as error:
                logger.warning(
                    "Skipping command '%s' (failed to load): %s",
                    "/".join(path + [name]),
                    error,
                )
                continue
            if subcommand is None:
                continue
            subpath = path + [name]
            if isinstance(subcommand, click.Group):
                _walk(subcommand, subpath)
            else:
                discovered.append(
                    DiscoveredCommand(tool_name(subpath), subpath, subcommand)
                )

    _walk(root_command, [])

    selected = []
    for entry in discovered:
        if "mcp" in entry.path:
            # Don't let the server expose itself (at any nesting level).
            continue
        if entry.tool_name in exclude:
            continue
        if (
            expose == "safe"
            and entry.tool_name in DEFAULT_UNSAFE
            and entry.tool_name not in include
        ):
            continue
        selected.append(entry)
    selected.sort(key=lambda entry: entry.tool_name)
    return selected


def _range_schema(param_type, json_type) -> dict:
    """Map a 'click.IntRange'/'click.FloatRange' to a JSON-schema fragment.

    Open bounds ('min_open'/'max_open') map to 'exclusiveMinimum'/
    'exclusiveMaximum' so the advertised schema matches what 'click' accepts.
    """
    schema: dict = {"type": json_type}
    if getattr(param_type, "min", None) is not None:
        if getattr(param_type, "min_open", False):
            schema["exclusiveMinimum"] = param_type.min
        else:
            schema["minimum"] = param_type.min
    if getattr(param_type, "max", None) is not None:
        if getattr(param_type, "max_open", False):
            schema["exclusiveMaximum"] = param_type.max
        else:
            schema["maximum"] = param_type.max
    return schema


def _scalar_schema(param_type) -> dict:
    """Map a 'click.ParamType' to a scalar JSON-schema fragment."""
    if isinstance(param_type, click.Choice):
        return {"type": "string", "enum": [str(c) for c in param_type.choices]}
    if isinstance(param_type, click.IntRange):
        return _range_schema(param_type, "integer")
    if isinstance(param_type, click.FloatRange):
        return _range_schema(param_type, "number")
    if isinstance(param_type, click.types.IntParamType):
        return {"type": "integer"}
    if isinstance(param_type, click.types.FloatParamType):
        return {"type": "number"}
    if isinstance(param_type, click.types.BoolParamType):
        return {"type": "boolean"}
    # Path, File, STRING, function callbacks, etc. all map to string. click
    # re-validates and converts the value when the command runs.
    return {"type": "string"}


def param_to_schema(param) -> dict:
    """Map a single 'click.Parameter' to a JSON-schema property."""
    schema: dict
    if getattr(param, "is_flag", False) or getattr(
        param, "is_bool_flag", False
    ):
        schema = {"type": "boolean"}
    else:
        schema = dict(_scalar_schema(param.type))
        nargs = getattr(param, "nargs", 1)
        multiple = getattr(param, "multiple", False)
        # A fixed 'nargs > 1' groups the values into a fixed-length array.
        if isinstance(nargs, int) and nargs > 1:
            schema = {
                "type": "array",
                "items": schema,
                "minItems": nargs,
                "maxItems": nargs,
            }
        # 'multiple' (or a variadic 'nargs == -1') wraps the per-invocation
        # value in an outer array, so a repeatable fixed-group option becomes
        # an array of fixed-length arrays.
        if multiple or nargs == -1:
            schema = {"type": "array", "items": schema}

    description = getattr(param, "help", None)
    if not description and isinstance(param, click.Argument):
        description = f"Positional argument '{param.name}'."
    if description:
        schema["description"] = description

    if not param.required and param.default is not None:
        try:
            # Only advertise a JSON round-trippable, non-empty default. click's
            # "no default" sentinel and other non-serializable defaults (such as
            # 'Path' objects, or array-like defaults whose '!=' comparison
            # raises 'ValueError') are simply omitted.
            default = param.default
            if default != () and default != []:
                json.dumps(default)
                schema["default"] = default
        except (TypeError, ValueError):
            pass
    return schema


def command_input_schema(command) -> dict:
    """Build the JSON-schema 'inputSchema' for a 'click' command."""
    properties = {}
    required = []
    for param in command.params:
        if not getattr(param, "expose_value", True):
            continue
        if getattr(param, "hidden", False):
            continue
        properties[param.name] = param_to_schema(param)
        if param.required:
            required.append(param.name)
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _primary_option(param) -> str:
    """Return the canonical (long) option string for an option parameter."""
    long_opts = [opt for opt in param.opts if opt.startswith("--")]
    if long_opts:
        return long_opts[0]
    return param.opts[0]


def arguments_to_argv(command, arguments) -> list:
    """Reconstruct a CLI argv tail from a tool-call 'arguments' dict.

    Options are emitted as '--name value' (flags as a bare '--flag', the false
    branch of a '--x/--no-x' flag as '--no-x'), repeatable options are repeated,
    and positional arguments are appended last. The resulting argv is parsed and
    re-validated by 'click' when the command runs.
    """
    options = []
    positionals = []
    for param in command.params:
        if not getattr(param, "expose_value", True):
            continue
        if param.name not in arguments:
            continue
        value = arguments[param.name]
        if value is None:
            continue

        if isinstance(param, click.Argument):
            if getattr(param, "nargs", 1) == 1 and not getattr(
                param, "multiple", False
            ):
                positionals.append(str(value))
            else:
                positionals.extend(str(item) for item in value)
            continue

        if getattr(param, "is_flag", False) or getattr(
            param, "is_bool_flag", False
        ):
            if value:
                options.append(_primary_option(param))
            elif param.secondary_opts:
                options.append(param.secondary_opts[0])
            continue

        option = _primary_option(param)
        if getattr(param, "count", False):
            # A 'count' option takes no value; it is repeated 'value' times
            # (e.g. '--verbose --verbose' for value == 2).
            options.extend([option] * int(value))
        elif getattr(param, "multiple", False):
            nargs = getattr(param, "nargs", 1)
            for item in value:
                if isinstance(nargs, int) and nargs > 1:
                    # A repeatable fixed-group option: emit the fixed group
                    # once per repetition (e.g. '--pair a b --pair c d').
                    options.append(option)
                    options.extend(str(sub) for sub in item)
                else:
                    options.extend([option, str(item)])
        elif isinstance(getattr(param, "nargs", 1), int) and param.nargs > 1:
            options.append(option)
            options.extend(str(item) for item in value)
        else:
            options.extend([option, str(value)])
    return options + positionals


def tool_description(command) -> str:
    """Build the MCP tool description from a command's help text."""
    text = command.help or command.short_help or ""
    # 'click' truncates the displayed help at the '\f' form-feed marker and uses
    # '\b' to mark paragraphs that should not be rewrapped. Strip both.
    text = text.split("\f", 1)[0]
    text = text.replace("\b\n", "").replace("\b", "")
    text = text.strip()
    if not text:
        text = f"Invoke the 'spectre {command.name}' CLI command."
    return text


def _truncate_output(text) -> str:
    """Cap a single stream at 'MAX_OUTPUT_CHARS', keeping its head and tail."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_OUTPUT_CHARS
    head = MAX_OUTPUT_CHARS // 2
    tail = MAX_OUTPUT_CHARS - head
    return (
        text[:head]
        + f"\n[... output truncated, {omitted} characters omitted ...]\n"
        + text[len(text) - tail :]
    )


def _format_result(returncode, stdout, stderr) -> str:
    """Format a command's exit code and captured output as MCP tool text."""
    parts = [f"[exit code {returncode}]"]
    out = _truncate_output((stdout or "").rstrip())
    err = _truncate_output((stderr or "").rstrip())
    if out:
        parts.append("--- stdout ---\n" + out)
    if err:
        parts.append("--- stderr ---\n" + err)
    if not out and not err:
        parts.append("(no output)")
    return "\n".join(parts)


def run_tool_subprocess(command_path, argv, timeout=None) -> str:
    """Run a 'spectre' command in a subprocess and return its captured output.

    This is the default execution mode: it fully isolates the command (including
    any executables it spawns) so their output is captured in a pipe and never
    leaks onto the MCP server's stdout, which is reserved for the protocol.
    """
    import subprocess

    command = [sys.executable, "-m", "spectre", *command_path, *argv]
    try:
        completed = subprocess.run(
            command,
            input="",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return (
            f"[timed out after {timeout} s running 'spectre "
            f"{' '.join(command_path + argv)}']\n"
            + _format_result(-1, stdout, stderr)
        )
    return _format_result(
        completed.returncode, completed.stdout, completed.stderr
    )


def run_tool_in_process(command_path, argv) -> str:
    """Run a 'spectre' command in-process via 'click.testing.CliRunner'.

    Faster than a subprocess (no interpreter startup), but Python-level stdout
    capture does not redirect the OS file descriptors inherited by child
    processes. Use for pure-Python commands; commands that spawn executables
    (e.g. 'run'/'schedule') should use 'run_tool_subprocess' so their output
    does not corrupt the MCP stdio channel. 'input=""' makes any interactive
    confirmation prompt fail fast instead of hanging.
    """
    from click.testing import CliRunner

    from spectre.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, [*command_path, *argv], input="", catch_exceptions=True
    )
    # 'result.output' is the combined stdout+stderr stream; use 'result.stdout'
    # so stderr is not duplicated in both sections. The 'try/except' guards
    # older 'click' versions where 'stderr' may not be separately captured.
    stdout = result.stdout or ""
    try:
        stderr = result.stderr
    except ValueError:
        stderr = ""
    text = _format_result(result.exit_code, stdout, stderr)
    if result.exception is not None and not isinstance(
        result.exception, SystemExit
    ):
        import traceback

        text += "\n--- exception ---\n" + "".join(
            traceback.format_exception(
                type(result.exception),
                result.exception,
                result.exception.__traceback__,
            )
        )
    return text


def _redirect_logging_to_stderr():
    """Send all logging to stderr so it can't corrupt the protocol on stdout."""
    import rich.console
    import rich.logging

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    handler = rich.logging.RichHandler(
        console=rich.console.Console(stderr=True),
        show_time=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)


@click.command(
    name="mcp",
    help=(
        "Run a Model Context Protocol (MCP) server that exposes the 'spectre'"
        " CLI commands as MCP tools over stdio.\n\nThe server auto-generates"
        " one tool per CLI command by introspecting the command tree, so MCP"
        " clients such as Claude Code can discover and invoke 'spectre'"
        " commands directly. By default a curated 'safe' subset is exposed"
        " (analysis, visualization, local execution, and data extraction);"
        " cluster-job and auto-resubmit pipeline commands are hidden until"
        " '--expose all'.\n\nThis command requires the optional 'mcp' Python"
        " package (Python >= 3.10). It is normally launched automatically by an"
        " MCP client via the repository's '.mcp.json', not run by hand."
    ),
)
@click.option(
    "--expose",
    type=click.Choice(["safe", "all"]),
    default="safe",
    show_default=True,
    help=(
        "Which commands to expose. 'safe' hides cluster-job/auto-resubmit"
        " commands; 'all' exposes every command."
    ),
)
@click.option(
    "--include",
    multiple=True,
    metavar="COMMAND",
    help=(
        "Expose this command even in 'safe' mode. The name is the tool name or"
        " CLI name (e.g. 'resubmit'). Can be given multiple times."
    ),
)
@click.option(
    "--exclude",
    multiple=True,
    metavar="COMMAND",
    help="Never expose this command. Can be given multiple times.",
)
@click.option(
    "--in-process",
    "in_process",
    is_flag=True,
    help=(
        "Run commands in-process instead of in a subprocess. Faster, but not"
        " safe for commands that spawn executables (their output can corrupt"
        " the stdio protocol). Prefer the default subprocess execution."
    ),
)
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help=(
        "Timeout in seconds for each command (subprocess execution only)."
        " Pass 0 to disable the timeout."
    ),
)
def mcp_command(expose, include, exclude, in_process, timeout):
    _rich_traceback_guard = True  # Hide traceback until here
    try:
        import anyio
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as error:
        raise click.ClickException(
            "The 'mcp' Python package is required to run the MCP server but"
            " could not be imported. Install it with 'pip install mcp'"
            " (requires Python >= 3.10).\n"
            f"Import error: {error}"
        )

    # stdout is the MCP protocol channel, so redirect all logging to stderr.
    _redirect_logging_to_stderr()

    from spectre.__main__ import cli

    discovered = discover_commands(
        cli, expose=expose, include=include, exclude=exclude
    )
    commands_by_name = {entry.tool_name: entry for entry in discovered}
    logger.info(
        "Starting spectre MCP server: exposing %d commands (expose=%s).",
        len(discovered),
        expose,
    )
    click.echo(
        (
            f"spectre MCP server: {len(discovered)} tools exposed "
            f"(expose={expose}, "
            f"execution={'in-process' if in_process else 'subprocess'})."
        ),
        err=True,
    )

    server = Server("spectre")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=entry.tool_name,
                description=tool_description(entry.command),
                inputSchema=command_input_schema(entry.command),
            )
            for entry in discovered
        ]

    # Disable server-side schema validation: 'click' re-validates every argument
    # and produces better error messages, so it is the single source of truth.
    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        entry = commands_by_name.get(name)
        if entry is None:
            return [
                types.TextContent(type="text", text=f"Unknown tool: '{name}'.")
            ]
        try:
            argv = arguments_to_argv(entry.command, arguments or {})
        except Exception as error:
            return [
                types.TextContent(
                    type="text",
                    text=f"Failed to build arguments for '{name}': {error}",
                )
            ]
        if in_process:
            text = await anyio.to_thread.run_sync(
                run_tool_in_process, entry.path, argv
            )
        else:
            text = await anyio.to_thread.run_sync(
                functools.partial(
                    # A timeout of 0 disables the limit (passed as None).
                    run_tool_subprocess,
                    entry.path,
                    argv,
                    timeout or None,
                )
            )
        return [types.TextContent(type="text", text=text)]

    async def serve():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    anyio.run(serve)


if __name__ == "__main__":
    mcp_command(help_option_names=["-h", "--help"])
