#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

import spectre.IO.H5 as spectre_h5


@click.command()
@click.argument(
    "h5_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument("output_file", required=False, type=click.File("w"))
def extract_input_source_from_h5_command(h5_file, output_file):
    """Extract input file from an H5 file

    Extract InputSource.yaml from the 'H5_FILE' and write it to the
    'OUTPUT_FILE', or print to stdout if `OUTPUT_FILE` is unspecified.
    """
    with spectre_h5.H5File(h5_file, "r") as open_file:
        input_source = open_file.input_source()

    if output_file:
        output_file.write(input_source)
    else:
        import rich.syntax

        syntax = rich.syntax.Syntax(
            input_source, lexer="yaml", theme="ansi_dark"
        )
        rich.print(syntax)


if __name__ == "__main__":
    extract_input_source_from_h5_command(help_option_names=["-h", "--help"])
