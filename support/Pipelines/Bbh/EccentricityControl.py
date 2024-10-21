# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import click
import pandas as pd

from spectre.Pipelines.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
    eccentricity_control_params_options,
)

logger = logging.getLogger(__name__)


def eccentricity_control(h5_files, id_input_file_path, **kwargs):
    """Eccentricity reduction post inspiral.

    This function can be called after the inspiral has run (see the 'Next'
    section of the Inspiral.yaml file).

    This function does the following:

    - Get the new orbital parameters by calling the function
      'eccentricity_control_params' in
      'spectre.Pipelines.EccentricityControl.EccentricityControl'.
    - Print the new orbital parameters in a tabular format. This will be updated
      to start a new inspiral with the updated parameters.

    Arguments:
      h5_file: file that contains the trajectory data
      id_input_file_path: path to the input file of the initial data run
      **kwargs: additional arguments to be forwarded to
        'eccentricity_control_params'.
    """
    # Find the current eccentricity and extract new parameters to put into
    # generate-id
    (
        eccentricity,
        ecc_std_dev,
        new_orbital_params,
    ) = eccentricity_control_params(h5_files, id_input_file_path, **kwargs)

    # Create DataFrame to display data in tabular format
    data = {
        "Attribute": [
            "Eccentricity",
            "Eccentricity error",
            "Updated Omega0",
            "Updated adot0",
        ],
        "Value": [
            eccentricity,
            ecc_std_dev,
            new_orbital_params["Omega0"],
            new_orbital_params["adot0"],
        ],
    }
    df = pd.DataFrame(data)
    # Print header line
    print("=" * 40)
    # Display table
    print(df.to_string(index=False))
    print("=" * 40)


@click.command(name="eccentricity-control", help=eccentricity_control.__doc__)
@eccentricity_control_params_options
def eccentricity_control_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    eccentricity_control(**kwargs)
