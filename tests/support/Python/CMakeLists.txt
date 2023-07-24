# Distributed under the MIT License.
# See LICENSE.txt for details.

if (BUILD_PYTHON_BINDINGS)
  # The SPECTRE_VERSION is inserted in __init__.py by cmake, so this test
  # doesn't work in PY_DEV_MODE
  if (NOT PY_DEV_MODE)
    add_test(NAME "support.Python.Cli"
      COMMAND ${CMAKE_BINARY_DIR}/bin/spectre --version)
    set_tests_properties(
      "support.Python.Cli" PROPERTIES
      PASS_REGULAR_EXPRESSION "${SPECTRE_VERSION}"
      LABELS "python")
  endif()

  add_test(NAME "support.Python.python-spectre"
    COMMAND ${CMAKE_BINARY_DIR}/bin/python-spectre -c
      "import spectre; print(spectre.__version__)")
  set_tests_properties(
    "support.Python.python-spectre" PROPERTIES
    PASS_REGULAR_EXPRESSION "${SPECTRE_VERSION}"
    LABELS "python")
endif()

spectre_add_python_bindings_test(
  "support.DirectoryStructure"
  Test_DirectoryStructure.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "support.Machines"
  Test_Machines.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "support.Python.Main"
  Test_Main.py
  "Python"
  None)

spectre_add_python_bindings_test(
  "support.Python.Schedule"
  Test_Schedule.py
  "Python"
  None)
