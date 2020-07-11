# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Cce)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Actions/ScriObserveInterpolated.cpp
  AnalyticBoundaryDataManager.cpp
  BoundaryData.cpp
  Equations.cpp
  GaugeTransformBoundaryData.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  NewmanPenrose.cpp
  PrecomputeCceDependencies.cpp
  ReadBoundaryDataH5.cpp
  ReducedWorldtubeModeRecorder.cpp
  ScriPlusValues.cpp
  SpecBoundaryData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticBoundaryDataManager.hpp
  BoundaryData.hpp
  BoundaryDataTags.hpp
  Equations.hpp
  GaugeTransformBoundaryData.hpp
  IntegrandInputSteps.hpp
  LinearOperators.hpp
  LinearSolve.hpp
  NewmanPenrose.hpp
  OptionTags.hpp
  PreSwshDerivatives.hpp
  PrecomputeCceDependencies.hpp
  ReadBoundaryDataH5.hpp
  ReceiveTags.hpp
  ReducedWorldtubeModeRecorder.hpp
  ScriPlusInterpolationManager.hpp
  ScriPlusValues.hpp
  SpecBoundaryData.hpp
  SwshDerivatives.hpp
  System.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  LinearSolver
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  IO
  LinearOperators
  Options
  Spectral
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(AnalyticSolutions)
add_subdirectory(Components)
add_subdirectory(Initialize)
add_subdirectory(InterfaceManagers)
