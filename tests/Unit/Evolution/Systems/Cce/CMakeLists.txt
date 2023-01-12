# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(AnalyticSolutions)

set(LIBRARY "Test_Cce")

set(LIBRARY_SOURCES
  InterfaceManagers/Test_GhLocalTimeStepping.cpp
  InterfaceManagers/Test_GhLockstep.cpp
  Test_AnalyticBoundaryDataManager.cpp
  Test_BoundaryData.cpp
  Test_BoundaryDataTags.cpp
  Test_DumpBondiSachsOnWorldtube.cpp
  Test_Equations.cpp
  Test_GaugeTransformBoundaryData.cpp
  Test_InitializeCce.cpp
  Test_LinearOperators.cpp
  Test_LinearSolve.cpp
  Test_NewmanPenrose.cpp
  Test_OptionTags.cpp
  Test_PreSwshDerivatives.cpp
  Test_PrecomputeCceDependencies.cpp
  Test_ScriPlusInterpolationManager.cpp
  Test_ScriPlusValues.cpp
  Test_SwshDerivatives.cpp
  Test_Tags.cpp
  Test_WorldtubeData.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  Cce
  CceHelpers
  DataStructures
  Observer
  Options
  Spectral
  GeneralRelativitySolutions
  )
