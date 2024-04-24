# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE ReduceCceWorldtube)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ReduceCceWorldtube.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  Cce
  GeneralRelativity
  Informer
  Printf
  Spectral
  SpinWeightedSphericalHarmonics
  )

if(BUILD_TESTING)
  add_dependencies(test-executables ${EXECUTABLE})
endif()
