# Distributed under the MIT License.
# See LICENSE.txt for details.

if (SpEC_FOUND)
  spectre_add_python_bindings_test(
    "support.Pipelines.EccentricityControl.EccentricityControlParams"
    Test_EccentricityControlParams.py
    "Python"
    None
    TIMEOUT 60)
  spectre_add_python_bindings_test(
    "support.Pipelines.EccentricityControl.InitialOrbitalParameters"
    Test_InitialOrbitalParameters.py
    "Python"
    None
    TIMEOUT 60)
endif()
