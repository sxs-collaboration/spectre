# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

SPECTRE_VERSION = "@SPECTRE_VERSION@"


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class Cli(click.MultiCommand):
    def list_commands(self, ctx):
        return []

    def get_command(self, ctx, name):
        raise NotImplementedError(f"The command '{name}' is not implemented.")


# Set up CLI entry point
@click.group(context_settings=dict(help_option_names=["-h", "--help"]),
             help=f"SpECTRE version: {SPECTRE_VERSION}",
             cls=Cli)
@click.version_option(version=SPECTRE_VERSION, message="%(version)s")
def cli():
    pass


if __name__ == "__main__":
    cli(prog_name="spectre")
