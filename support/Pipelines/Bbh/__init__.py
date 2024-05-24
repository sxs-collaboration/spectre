# Distributed under the MIT License.
# See LICENSE.txt for details.

import click


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class Bbh(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "find-horizon",
            "generate-id",
            "start-inspiral",
            "start-ringdown",
        ]

    def get_command(self, ctx, name):
        if name == "find-horizon":
            from .FindHorizon import find_horizon_command

            return find_horizon_command
        if name == "generate-id":
            from .InitialData import generate_id_command

            return generate_id_command
        elif name == "start-inspiral":
            from .Inspiral import start_inspiral_command

            return start_inspiral_command
        elif name == "start-ringdown":
            from .Ringdown import start_ringdown_command

            return start_ringdown_command

        available_commands = " " + "\n ".join(self.list_commands(ctx))
        raise click.UsageError(
            f"The command '{name}' is not implemented. "
            f"Available commands are:\n{available_commands}"
        )


@click.group(name="bbh", cls=Bbh)
def bbh_pipeline():
    """Pipeline for binary black hole simulations."""
    pass


if __name__ == "__main__":
    bbh_pipeline(help_option_names=["-h", "--help"])
