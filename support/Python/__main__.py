# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich.traceback
import yaml

import spectre
from spectre.support.CliExceptions import RequiredChoiceError
from spectre.support.Logging import configure_logging
from spectre.support.Machines import UnknownMachineError, this_machine

logger = logging.getLogger(__name__)


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class Cli(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "bbh",
            "clean-output",
            "combine-h5",
            "delete-subfiles",
            "eccentricity-control",
            "extend-connectivity",
            "extract-dat",
            "extract-input",
            "find-radial-surface",
            "generate-tetrahedral-connectivity",
            "generate-xdmf",
            "interpolate-to-mesh",
            "interpolate-to-points",
            "plot",
            "render-1d",
            "render-3d",
            "resubmit",
            "run-next",
            "schedule",
            "simplify-traces",
            "status",
            "transform-volume-data",
            "validate",
        ]

    def get_command(self, ctx, name):
        if name == "bbh":
            from spectre.Pipelines.Bbh import bbh_pipeline

            return bbh_pipeline
        if name == "clean-output":
            from spectre.tools.CleanOutput import clean_output_command

            return clean_output_command
        elif name == "combine-h5":
            from spectre.IO.H5.CombineH5 import combine_h5_command

            return combine_h5_command
        elif name == "delete-subfiles":
            from spectre.IO.H5.DeleteSubfiles import delete_subfiles_command

            return delete_subfiles_command

        elif name == "eccentricity-control":
            from spectre.Pipelines.EccentricityControl import (
                eccentricity_control_command,
            )

            return eccentricity_control_command
        elif name == "extend-connectivity":
            from spectre.IO.H5.ExtendConnectivityData import (
                extend_connectivity_data_command,
            )

            return extend_connectivity_data_command
        elif name == "extract-dat":
            from spectre.IO.H5.ExtractDatFromH5 import extract_dat_command

            return extract_dat_command
        elif name == "extract-input":
            from spectre.IO.H5.ExtractInputSourceYamlFromH5 import (
                extract_input_source_from_h5_command,
            )

            return extract_input_source_from_h5_command
        elif name == "find-radial-surface":
            from spectre.SurfaceFinder.FindRadialSurface import (
                find_radial_surface_command,
            )

            return find_radial_surface_command
        elif name == "generate-tetrahedral-connectivity":
            from spectre.Visualization.GenerateTetrahedralConnectivity import (
                generate_tetrahedral_connectivity_command,
            )

            return generate_tetrahedral_connectivity_command
        elif name == "generate-xdmf":
            from spectre.Visualization.GenerateXdmf import generate_xdmf_command

            return generate_xdmf_command
        elif name in ["interpolate-to-points", "interpolate-to-coords"]:
            from spectre.IO.Exporter.InterpolateToPoints import (
                interpolate_to_points_command,
            )

            return interpolate_to_points_command
        elif name == "interpolate-to-mesh":
            from spectre.Visualization.InterpolateToMesh import (
                interpolate_to_mesh_command,
            )

            return interpolate_to_mesh_command
        elif name == "plot":
            from spectre.Visualization import plot_command

            return plot_command
        elif name == "render-1d":
            from spectre.Visualization.Render1D import render_1d_command

            return render_1d_command
        elif name == "render-3d":
            from spectre.Visualization.Render3D import render_3d_command

            return render_3d_command
        elif name == "resubmit":
            from spectre.support.Resubmit import resubmit_command

            return resubmit_command
        elif name == "run-next":
            from spectre.support.RunNext import run_next_command

            return run_next_command
        elif name in ["schedule", "run"]:
            from spectre.support.Schedule import schedule_command

            return schedule_command
        elif name == "simplify-traces":
            from spectre.tools.CharmSimplifyTraces import (
                simplify_traces_command,
            )

            return simplify_traces_command
        elif name == "status":
            from spectre.tools.Status import status_command

            return status_command
        elif name in ["transform-volume-data", "transform-vol"]:
            from spectre.Visualization.TransformVolumeData import (
                transform_volume_data_command,
            )

            return transform_volume_data_command
        elif name == "validate":
            from spectre.tools.ValidateInputFile import (
                validate_input_file_command,
            )

            return validate_input_file_command
        raise RequiredChoiceError(
            f"The command '{name}' is not implemented.",
            choices=self.list_commands(ctx),
        )


def print_machine(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    try:
        machine = this_machine()
        click.echo(machine.Name)
        ctx.exit(1)
    except UnknownMachineError as exc:
        click.echo(exc)
        ctx.exit()


def read_config_file(ctx, param, config_file):
    if not config_file:
        return
    config_file = os.path.expanduser(config_file)
    if not os.path.exists(config_file):
        return
    with open(config_file, "r") as open_config_file:
        config = yaml.safe_load(open_config_file)
    ctx.default_map = config


# Set up CLI entry point
@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help=f"SpECTRE version: {spectre.__version__}",
    cls=Cli,
)
@click.version_option(version=spectre.__version__, message="%(version)s")
@click.option(
    "--machine",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=print_machine,
    help="Show the machine we're running on and exit.",
)
@click.option(
    "--debug",
    "log_level",
    flag_value=logging.DEBUG,
    help="Enable debug logging.",
)
@click.option(
    "--silent",
    "log_level",
    flag_value=logging.CRITICAL,
    help="Disable all logging.",
)
@click.option(
    "--build-dir",
    "-b",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help=(
        "Prepend a build directory to the PATH "
        "so subprocesses can find executables in it. "
        "Without this option, executables are found in the "
        "current PATH, and fall back to the build directory "
        "in which this Python script is installed."
    ),
)
@click.option(
    "--profile",
    is_flag=True,
    help=(
        "Enable profiling. "
        "Expect slower execution due to profiling overhead. "
        "A summary of the results is printed to the terminal. "
        "Use the '--output-profile' option to write the results "
        "to a file."
    ),
)
@click.option(
    "--output-profile",
    type=click.Path(writable=True),
    help=(
        "Write profiling results to a file. "
        "The file can be opened by profiling visualization tools "
        "such as 'pstats' or 'gprof2dot'. "
        "See the Python 'cProfile' docs for details."
    ),
)
@click.option(
    "-c",
    "--config-file",
    type=click.Path(file_okay=True, dir_okay=False),
    callback=read_config_file,
    default="~/.config/spectre.yaml",
    show_default=True,
    is_eager=True,
    expose_value=False,
    envvar="SPECTRE_CONFIG_FILE",
    help=(
        "Configuration file in YAML format. Can provide defaults "
        "for command-line options and additional configuration. "
        "To specify options for subcommands, list them in a "
        "section with the same name as the subcommand. "
        "All options that are listed in the help string for a "
        "subcommand are supported. Unless otherwise specified in "
        "the help string, use the name of the option with dashes "
        "replaced by underscores. Example:\n\n"
        "\b\n"
        "status:\n"
        "  starttime: now-2days\n"
        "  state_styles:\n"
        "    RUNNING: blink\n"
        "plot:\n"
        "  dat:\n"
        "    stylesheet: path/to/stylesheet.mplstyle\n\n"
        "The path of the config file can also be specified by "
        "setting the 'SPECTRE_CONFIG_FILE' environment variable."
    ),
)
def cli(log_level, build_dir, profile, output_profile):
    configure_logging(log_level=log_level)
    # Format tracebacks with rich
    # - Suppress traceback entries from modules that we don't care about
    rich.traceback.install(
        show_locals=log_level is None or log_level <= logging.DEBUG,
        extra_lines=(
            3 if log_level is None or log_level <= logging.DEBUG else 0
        ),
        suppress=[click],
    )
    # Add the build directory to the PATH so subprocesses can find executables.
    # - We respect the user's PATH and only add a build directory if it was
    #   explicitly specified on the command line.
    if build_dir:
        os.environ["PATH"] = (
            os.path.join(build_dir, "bin") + ":" + os.environ["PATH"]
        )
    # - We fall back to executables in this file's build directory as a last
    #   resort.
    default_bin_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    os.environ["PATH"] = os.environ["PATH"] + ":" + default_bin_dir

    # Configure profiling
    if profile:
        import atexit
        import cProfile
        import pstats
        from io import StringIO

        logger.info(
            "Profiling is enabled. Expect slower execution due to "
            "profiling overhead."
        )
        profiler = cProfile.Profile()
        profiler.enable()

        # Output profiling result at exit
        def complete_profiling():
            profiler.disable()
            if output_profile:
                # Write to file
                profiler.dump_stats(output_profile)
            else:
                # Print to terminal
                import rich.console

                console = rich.console.Console()
                console.rule(
                    "[bold]Profiling Result", style="black", align="center"
                )
                profile_report = StringIO()
                stats = pstats.Stats(
                    profiler, stream=profile_report
                ).sort_stats(pstats.SortKey.TIME)
                stats.print_stats(20)
                console.print(profile_report.getvalue())

        atexit.register(complete_profiling)


if __name__ == "__main__":
    cli(prog_name="spectre")
