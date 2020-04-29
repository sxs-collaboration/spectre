# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(AnalyticSolutions)

set(LIBRARY Cce)

set(LIBRARY_SOURCES
  BoundaryData.cpp
  Equations.cpp
  GaugeTransformBoundaryData.cpp
  Initialize/InitializeJ.cpp
  Initialize/InverseCubic.cpp
  Initialize/NoIncomingRadiation.cpp
  Initialize/ZeroNonSmooth.cpp
  InterfaceManagers/GhLockstepInterfaceManager.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  NewmanPenrose.cpp
  PrecomputeCceDependencies.cpp
  ReadBoundaryDataH5.cpp
  ReducedWorldtubeModeRecorder.cpp
  SpecBoundaryData.cpp
  ScriPlusValues.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  LinearSolver

  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  IO
  LinearOperators
  Options
  Spectral
  Utilities
  )
