# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.InitialData"
  Test_InitialData.py
  "Python"
  None)

# Disable if compiler is GCC because for some reason it doesn't trap the
# exception thrown by the test. Can be re-enabled once the test can find a
# horizon in some data, so the exception trapping is not needed anymore.
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  spectre_add_python_bindings_test(
    "support.Pipelines.Bbh.FindHorizon"
    Test_FindHorizon.py
    "Python"
    None)
endif()

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.Inspiral"
  Test_Inspiral.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.Ringdown"
  Test_Ringdown.py
  "Python"
  None)

if (BUILD_PYTHON_BINDINGS)
  set_tests_properties(
    "support.Pipelines.Bbh.InitialData"
    PROPERTIES TIMEOUT 60)
  set_tests_properties(
    "support.Pipelines.Bbh.Inspiral"
    PROPERTIES TIMEOUT 60)
  set_tests_properties(
    "support.Pipelines.Bbh.Ringdown"
    PROPERTIES TIMEOUT 60)
endif()
