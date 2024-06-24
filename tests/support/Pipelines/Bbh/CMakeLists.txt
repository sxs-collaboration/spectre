# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.FindHorizon"
  Test_FindHorizon.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.InitialData"
  Test_InitialData.py
  "Python"
  None)

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
    "support.Pipelines.Bbh.FindHorizon"
    PROPERTIES TIMEOUT 20)
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
