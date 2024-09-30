# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_FILES
  __init__.py
  GenerateTetrahedralConnectivity.py
  GenerateXdmf.py
  InterpolateToMesh.py
  OpenVolfiles.py
  Plot.py
  PlotAlongLine.py
  PlotCce.py
  PlotControlSystem.py
  PlotDatFile.py
  PlotEllipticConvergence.py
  PlotMemoryMonitors.py
  PlotPowerMonitors.py
  PlotSizeControl.py
  PlotSlice.py
  PlotTrajectories.py
  plots.mplstyle
  ReadH5.py
  ReadInputFile.py
  Render1D.py
  TransformVolumeData.py
)

add_subdirectory(Render3D)
