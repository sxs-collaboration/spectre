# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import logging
import rich.logging
import rich.traceback
import spectre
from spectre.support.Machines import this_machine, UnknownMachineError

logger = logging.getLogger(__name__)


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class Cli(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "clean-output",
            "delete-subfiles",
            "extract-dat",
            "extract-input",
            "generate-xdmf",
            "interpolate-to-coords",
            "interpolate-to-mesh",
            "plot-dat",
            "plot-power-monitors",
            "render-1d",
            "simplify-traces",
            "transform-volume-data",
        ]

    def get_command(self, ctx, name):
        if name == "clean-output":
            from spectre.tools.CleanOutput import clean_output_command
            return clean_output_command
        elif name == "delete-subfiles":
            from spectre.IO.H5.DeleteSubfiles import delete_subfiles_command
            return delete_subfiles_command
        elif name == "extract-dat":
            from spectre.IO.H5.ExtractDatFromH5 import extract_dat_command
            return extract_dat_command
        elif name == "extract-input":
            from spectre.IO.H5.ExtractInputSourceYamlFromH5 import (
                extract_input_source_from_h5_command)
            return extract_input_source_from_h5_command
        elif name == "generate-xdmf":
            from spectre.Visualization.GenerateXdmf import (
                generate_xdmf_command)
            return generate_xdmf_command
        elif name == "interpolate-to-coords":
            from spectre.Visualization.InterpolateToCoords import (
                interpolate_to_coords_command)
            return interpolate_to_coords_command
        elif name == "interpolate-to-mesh":
            from spectre.Visualization.InterpolateToMesh import (
                interpolate_to_mesh_command)
            return interpolate_to_mesh_command
        elif name == "plot-dat":
            from spectre.Visualization.PlotDatFile import plot_dat_command
            return plot_dat_command
        elif name == "plot-power-monitors":
            from spectre.Visualization.PlotPowerMonitors import (
                plot_power_monitors_command)
            return plot_power_monitors_command
        elif name == "render-1d":
            from spectre.Visualization.Render1D import render_1d_command
            return render_1d_command
        elif name == "simplify-traces":
            from spectre.tools.CharmSimplifyTraces import (
                simplify_traces_command)
            return simplify_traces_command
        elif name in ["transform-volume-data", "transform-vol"]:
            from spectre.Visualization.TransformVolumeData import (
                transform_volume_data_command)
            return transform_volume_data_command

        available_commands = " " + "\n ".join(self.list_commands(ctx))
        raise click.UsageError(
            f"The command '{name}' is not implemented. "
            f"Available commands are:\n{available_commands}")


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


# Set up CLI entry point
@click.group(context_settings=dict(help_option_names=["-h", "--help"]),
             help=f"SpECTRE version: {spectre.__version__}",
             cls=Cli)
@click.version_option(version=spectre.__version__, message="%(version)s")
@click.option('--machine',
              is_flag=True,
              expose_value=False,
              is_eager=True,
              callback=print_machine,
              help="Show the machine we're running on and exit.")
@click.option('--debug',
              'log_level',
              flag_value=logging.DEBUG,
              help=("Enable debug logging."))
@click.option('--silent',
              'log_level',
              flag_value=logging.CRITICAL,
              help=("Disable all logging."))
@click.option('--profile',
              is_flag=True,
              help=("Enable profiling. "
                    "Expect slower execution due to profiling overhead. "
                    "A summary of the results is printed to the terminal. "
                    "Use the '--output-profile' option to write the results "
                    "to a file."))
@click.option('--output-profile',
              type=click.Path(writable=True),
              help=("Write profiling results to a file. "
                    "The file can be opened by profiling visualization tools "
                    "such as 'pstats' or 'gprof2dot'. "
                    "See the Python 'cProfile' docs for details."))
def cli(log_level, profile, output_profile):
    if log_level is None:
        log_level = logging.INFO
    # Configure logging
    logging.basicConfig(level=log_level,
                        format="%(message)s",
                        datefmt="[%X]",
                        handlers=[rich.logging.RichHandler()])
    # Format tracebacks with rich
    # - Suppress traceback entries from modules that we don't care about
    rich.traceback.install(
        show_locals=log_level <= logging.DEBUG,
        extra_lines=(3 if log_level <= logging.DEBUG else 0),
        suppress=[click])

    # Configure profiling
    if profile:
        import atexit
        import cProfile
        import pstats
        from io import StringIO

        logger.info("Profiling is enabled. Expect slower execution due to "
                    "profiling overhead.")
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
                console.rule("[bold]Profiling Result",
                             style="black",
                             align="center")
                profile_report = StringIO()
                stats = pstats.Stats(profiler,
                                     stream=profile_report).sort_stats(
                                         pstats.SortKey.TIME)
                stats.print_stats(20)
                console.print(profile_report.getvalue())

        atexit.register(complete_profiling)


if __name__ == "__main__":
    cli(prog_name="spectre")
