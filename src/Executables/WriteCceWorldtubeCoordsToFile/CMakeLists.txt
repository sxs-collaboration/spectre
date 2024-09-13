# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE WriteCceWorldtubeCoordsToFile)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  WriteCceWorldtubeCoordsToFile.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  Printf
  SphericalHarmonics
  SphericalHarmonicsIO
  )

if(BUILD_TESTING)
  add_dependencies(test-executables ${EXECUTABLE})
endif()
