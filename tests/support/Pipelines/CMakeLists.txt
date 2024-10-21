# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  Pipelines
  MODULE_PATH "testing"
  PYTHON_FILES
  MockBinaryData.py
)

add_subdirectory(Bbh)
add_subdirectory(EccentricityControl)
