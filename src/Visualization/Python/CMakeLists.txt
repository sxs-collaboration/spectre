# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  __init__.py
  GenerateXdmf.py
  InterpolateToCoords.py
  InterpolateToMesh.py
  PlotDatFile.py
  PlotPowerMonitors.py
  plots.mplstyle
  ReadH5.py
  ReadInputFile.py
  Render1D.py
  TransformVolumeData.py
)
