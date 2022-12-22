# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  GenerateXdmf.py
  PlotDatFile.py
  ReadH5.py
  Render1D.py
  PYTHON_EXECUTABLES
  InterpolateVolumeData.py
)
