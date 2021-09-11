# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
        "Unit.Spectral.Python.CollocationPoints"
        Test_CollocationPoints.py
        "Unit;Spectral;Python"
        PySpectral)

spectre_add_python_bindings_test(
        "Unit.Spectral.Python.Mesh"
        Test_Mesh.py
        "Unit;Spectral;Python"
        PySpectral)
