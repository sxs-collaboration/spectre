# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "tools.CharmSimplifyTraces"
  Test_CharmSimplifyTraces.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "tools.CleanOutput"
  Test_CleanOutput.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "tools.Status"
  Test_Status.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "tools.ValidateInputFile"
  Test_ValidateInputFile.py
  "Python"
  None)

if(${BUILD_PYTHON_BINDINGS})
  # Test is a bit slow because it runs an executable
  set_tests_properties(
    "tools.ValidateInputFile"
    PROPERTIES
    TIMEOUT 10
    )
endif()
