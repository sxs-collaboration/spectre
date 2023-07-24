# Distributed under the MIT License.
# See LICENSE.txt for details.

# The HDF5_REPACK_EXECUTABLE is inserted in DeleteSubfiles.py by cmake, so this
# test doesn't work in PY_DEV_MODE
if (NOT PY_DEV_MODE)
  spectre_add_python_bindings_test(
    "Unit.IO.H5.Python.DeleteSubfiles"
    Test_DeleteSubfiles.py
    "unit;IO;H5;python"
    PyH5)
endif()

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python.ExtendConnectivity"
  Test_ExtendConnectivity.py
  "unit;IO;H5;python"
  PyH5)

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python.ExtractDatFromH5"
  Test_ExtractDatFromH5.py
  "unit;IO;H5;python"
  PyH5
  )

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python.ExtractInputSourceYAMLFromH5"
  Test_ExtractInputSourceYamlFromH5.py
  "unit;IO;H5;python"
  PyH5)

spectre_add_python_bindings_test(
  "Unit.IO.H5.Python.IterElements"
  Test_IterElements.py
  "unit;IO;H5;python"
  PyH5)


