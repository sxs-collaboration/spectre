# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Cce)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AnalyticBoundaryDataManager.cpp
  BoundaryData.cpp
  Equations.cpp
  ExtractionRadius.cpp
  GaugeTransformBoundaryData.cpp
  KleinGordonSource.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  NewmanPenrose.cpp
  PrecomputeCceDependencies.cpp
  ScriPlusValues.cpp
  SpecBoundaryData.cpp
  WorldtubeBufferUpdater.cpp
  WorldtubeDataManager.cpp
  WorldtubeModeRecorder.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticBoundaryDataManager.hpp
  BoundaryData.hpp
  BoundaryDataTags.hpp
  Equations.hpp
  ExtractionRadius.hpp
  GaugeTransformBoundaryData.hpp
  IntegrandInputSteps.hpp
  KleinGordonSystem.hpp
  LinearOperators.hpp
  LinearSolve.hpp
  NewmanPenrose.hpp
  OptionTags.hpp
  PreSwshDerivatives.hpp
  PrecomputeCceDependencies.hpp
  ReceiveTags.hpp
  ScriPlusInterpolationManager.hpp
  ScriPlusValues.hpp
  SpecBoundaryData.hpp
  SwshDerivatives.hpp
  System.hpp
  KleinGordonSource.hpp
  KleinGordonSystem.hpp
  Tags.hpp
  WorldtubeBufferUpdater.hpp
  WorldtubeDataManager.hpp
  WorldtubeModeRecorder.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  LinearSolver
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  GeneralizedHarmonic
  GeneralRelativity
  H5
  Interpolation
  LinearOperators
  Observer
  Options
  ParallelInterpolation
  Printf
  Serialization
  Spectral
  SpinWeightedSphericalHarmonics
  Utilities
  INTERFACE
  SystemUtilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

add_subdirectory(Actions)
add_subdirectory(AnalyticSolutions)
add_subdirectory(Callbacks)
add_subdirectory(Components)
add_subdirectory(Events)
add_subdirectory(Initialize)
add_subdirectory(InterfaceManagers)
