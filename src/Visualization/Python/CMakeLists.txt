# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  __init__.py
  GenerateXdmf.py
  InterpolateToMesh.py
  PlotAlongLine.py
  PlotDatFile.py
  PlotPowerMonitors.py
  PlotSizeControl.py
  plots.mplstyle
  ReadH5.py
  ReadInputFile.py
  Render1D.py
  TransformVolumeData.py
)

add_subdirectory(Render3D)
