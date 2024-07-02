# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_add_python_bindings_test(
  "support.Pipelines.EccentricityControl.EccentricityControl"
  Test_EccentricityControl.py
  "Python"
  None)

if (SpEC_FOUND)
  spectre_add_python_bindings_test(
    "support.Pipelines.EccentricityControl.InitialOrbitalParameters"
    Test_InitialOrbitalParameters.py
    "Python"
    None)

  if (BUILD_PYTHON_BINDINGS)
    set_tests_properties(
      "support.Pipelines.EccentricityControl.EccentricityControl"
      PROPERTIES TIMEOUT 60)
    set_tests_properties(
      "support.Pipelines.EccentricityControl.InitialOrbitalParameters"
      PROPERTIES TIMEOUT 60)
  endif()
endif()
