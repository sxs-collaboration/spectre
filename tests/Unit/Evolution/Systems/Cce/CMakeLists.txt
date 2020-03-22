# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)

set(LIBRARY "Test_Cce")

set(LIBRARY_SOURCES
  InterfaceManagers/Test_GhLockstepInterfaceManager.cpp
  Test_BoundaryData.cpp
  Test_Equations.cpp
  Test_GaugeTransformBoundaryData.cpp
  Test_InitializeCce.cpp
  Test_LinearOperators.cpp
  Test_LinearSolve.cpp
  Test_NewmanPenrose.cpp
  Test_OptionTags.cpp
  Test_PreSwshDerivatives.cpp
  Test_PrecomputeCceDependencies.cpp
  Test_ReadBoundaryDataH5.cpp
  Test_ScriPlusInterpolationManager.cpp
  Test_ScriPlusValues.cpp
  Test_SwshDerivatives.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/"
  "${LIBRARY_SOURCES}"
  "Boost::boost;Cce;CceHelpers;GeneralRelativitySolutions"
  )
