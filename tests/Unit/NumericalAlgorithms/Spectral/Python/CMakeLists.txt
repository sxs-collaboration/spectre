# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
        "Unit.Spectral.Python.LogicalCoordinates"
        Test_LogicalCoordinates.py
        "Unit;Spectral;Python"
        PySpectral)

spectre_add_python_bindings_test(
        "Unit.Spectral.Python.Mesh"
        Test_Mesh.py
        "Unit;Spectral;Python"
        PySpectral)

spectre_add_python_bindings_test(
        "Unit.Spectral.Python.Spectral"
        Test_Spectral.py
        "Unit;Spectral;Python"
        PySpectral)
