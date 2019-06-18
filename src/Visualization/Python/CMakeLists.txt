# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  Visualization/Python/GenerateXdmf.py
  Visualization/Python/Render1D.py
  )

spectre_python_add_executable(
  GenerateXdmf
  Visualization/GenerateXdmf.py
)

spectre_python_add_executable(
  Render1D
  Visualization/Render1D.py
)
