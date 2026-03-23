# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  __init__.py
  ElementIdToParaview.py
  GenerateTetrahedralConnectivity.py
  GenerateXdmf.py
  Plot.py
  PlotAlongLine.py
  PlotCce.py
  PlotControlSystem.py
  PlotDatFile.py
  PlotEccentricityControl.py
  PlotEllipticConvergence.py
  PlotMemoryMonitors.py
  PlotPowerMonitors.py
  PlotSizeControl.py
  PlotSlice.py
  PlotTrajectories.py
  plots.mplstyle
  ReadInputFile.py
  Render1D.py
)

add_subdirectory(Render3D)
