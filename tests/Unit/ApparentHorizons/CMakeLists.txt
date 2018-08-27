# Distributed under the MIT License.
# See LICENSE.txt for details.

add_executable(
  Test_HorizonManager
  Test_HorizonManager.cpp
)

add_dependencies(
  Test_HorizonManager
  module_ConstGlobalCache
  module_Main
 )

target_link_libraries(
  Test_HorizonManager
  ApparentHorizons
  DataStructures
  DomainCreators
  CoordinateMaps
  EinsteinSolutions
  ErrorHandling
  GeneralRelativity
  Informer
  Interpolation
  LinearAlgebra
  Time
  Utilities
  ${SPECTRE_LIBRARIES}
)

add_dependencies(test-executables Test_HorizonManager)

add_test(NAME "\"Integration.HorizonManager\""
  COMMAND ${CMAKE_BINARY_DIR}/bin/Test_HorizonManager +p2 --input-file ${CMAKE_SOURCE_DIR}/tests/Unit/ApparentHorizons/Test_HorizonManager.input)

set_tests_properties(
  "\"Integration.HorizonManager\"" PROPERTIES
  TIMEOUT 30
  LABELS "integration"
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

set(LIBRARY "Test_ApparentHorizons")

set(LIBRARY_SOURCES
  StrahlkorperGrTestHelpers.cpp
  Test_FastFlow.cpp
  Test_SpherepackIterator.cpp
  Test_Strahlkorper.cpp
  Test_StrahlkorperGr.cpp
  Test_Tags.cpp
  Test_YlmSpherepack.cpp
  YlmTestFunctions.cpp
  )

add_test_library(
  ${LIBRARY}
  "ApparentHorizons"
  "${LIBRARY_SOURCES}"
  "ApparentHorizons;EinsteinSolutions;RootFinding;Utilities"
  )
