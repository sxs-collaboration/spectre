# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  tools
  PYTHON_FILES
  CharmSimplifyTraces.py
  CleanOutput.py
  ValidateInputFile.py
)

add_subdirectory(Status)
