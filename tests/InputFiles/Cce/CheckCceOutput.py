# Distributed under the MIT License.
# See LICENSE.txt for details.

# Checks the output of a CharacteristicExtract run against reference data.
# This script is shipped in the CceExecutables release tarball (see
# docs/Tutorials/CCE.md) and is run by the DeployStaticExecutables workflow.
# It is meant to be run from the root of the tarball, after running the
# CharacteristicExtract.yaml input file that sits there, so that the reference
# data is at Tests/CharacteristicExtractReduction_Expected.h5 and the freshly
# written output is at CharacteristicExtractReduction.h5.

import sys

import h5py
import numpy as np

expected_file = h5py.File(
    "Tests/CharacteristicExtractReduction_Expected.h5", "r"
)
new_file = h5py.File("CharacteristicExtractReduction.h5", "r")

vars = ["News", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4", "Strain"]
abs_tol = 3e-7
rel_tol = 3e-7

all_okay = True

for var in vars:
    # This used to increase the Psi0 tolerance but it was never
    # documented why that was needed. After a bugfix in the Psi0
    # computation and the second-order initial data it is not clear
    # this tolerance bump is still necessary. We are leaving the code
    # here just in case it is needed again in the future.
    #
    # local_abs_tol = abs_tol * 1e4 if var == "Psi0" else abs_tol
    # local_rel_tol = rel_tol * 1e4 if var == "Psi0" else rel_tol
    local_abs_tol = abs_tol
    local_rel_tol = rel_tol
    print("Testing ", var)
    expected_data = np.asarray(expected_file["/SpectreR0200.cce/" + var])
    new_data = np.asarray(new_file["/SpectreR0200.cce/" + var])
    error = (expected_data - new_data) / (
        np.maximum(np.abs(expected_data), np.abs(new_data)) * rel_tol + abs_tol
    )
    print("Time differences: ", np.max(error[:, 0]) * abs_tol)
    print("Variable differences: ", np.max(expected_data - new_data))
    print("Variable error: ", np.max(error[:, 1:]), "\n\n")

    if np.max(error[:, 1:]) > 1.0:
        all_okay = False

if all_okay:
    print("SUCCESS: The CCE output is as expected! Yay!")
else:
    print(
        "ERROR: The CCE output differes by more than expected. "
        "Please file an issue at github.com/sxs-collaboration/spectre/issues"
    )
    sys.exit(1)
