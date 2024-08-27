# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_add_python_bindings_test(
  "support.Pipelines.EccentricityControl.EccentricityControl"
  Test_EccentricityControl.py
  "Python"
  None
  TIMEOUT 60)

if (SpEC_FOUND)
  spectre_add_python_bindings_test(
    "support.Pipelines.EccentricityControl.InitialOrbitalParameters"
    Test_InitialOrbitalParameters.py
    "Python"
    None
    TIMEOUT 60)
endif()
