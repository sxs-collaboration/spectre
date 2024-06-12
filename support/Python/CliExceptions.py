# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import rich


class RequiredChoiceError(click.UsageError):
    """Error raised in the CLI when a required choice is not specified.

    Shows the list of available choices alongside the 'click.UsageError'.
    """

    def __init__(self, message, choices):
        super().__init__(message)
        self.choices = choices

    def format_message(self):
        import rich.columns
        import rich.console

        # Format available choices as columns for better readability
        console = rich.console.Console()
        with console.capture() as capture:
            console.print(rich.columns.Columns(self.choices))
        choices_cols = capture.get()

        return f"{self.message} Available choices:\n{choices_cols.strip()}"
