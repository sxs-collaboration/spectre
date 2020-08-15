# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ApparentHorizons")

set(LIBRARY_SOURCES
  Test_ApparentHorizonFinder.cpp
  Test_ChangeCenterOfStrahlkorper.cpp
  Test_ComputeItems.cpp
  Test_FastFlow.cpp
  Test_SpherepackIterator.cpp
  Test_Strahlkorper.cpp
  Test_StrahlkorperGr.cpp
  Test_Tags.cpp
  Test_YlmSpherepack.cpp
  )

add_test_library(
  ${LIBRARY}
  "ApparentHorizons"
  "${LIBRARY_SOURCES}"
  "ApparentHorizons;ApparentHorizonsHelpers;GeneralRelativitySolutions;Options;RootFinding;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
