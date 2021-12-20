# Distributed under the MIT License.
# See LICENSE.txt for details.

if (BUILD_PYTHON_BINDINGS)
  add_test(NAME "support.Python.main"
    COMMAND ${CMAKE_BINARY_DIR}/bin/spectre --version)
  set_tests_properties(
    "support.Python.main" PROPERTIES
    PASS_REGULAR_EXPRESSION "${SPECTRE_VERSION}"
    LABELS "python")
endif()

spectre_add_python_bindings_test(
  "support.Machines"
  Test_Machines.py
  "Machines"
  None)
