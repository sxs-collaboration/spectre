# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_add_python_bindings_test(
  "support.Pipelines.EccentricityControl.EccentricityControl"
  Test_EccentricityControl.py
  "Python"
  None
  TIMEOUT 60)

# Disable this test if using jemalloc as a shared library, because Julia
# doesn't like the 'LD_PRELOAD' trick to load jemalloc for Pybindings.
if (NOT "${JEMALLOC_LIB_TYPE}" STREQUAL SHARED)
  spectre_add_python_bindings_test(
    "support.Pipelines.EccentricityControl.InitialOrbitalParameters"
    Test_InitialOrbitalParameters.py
    "Python"
    None
    TIMEOUT 60)
endif()
