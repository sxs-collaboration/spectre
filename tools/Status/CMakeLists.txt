# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  Status
  MODULE_PATH "tools"
  PYTHON_FILES
  __init__.py
  Status.py
)

add_subdirectory(ExecutableStatus)
