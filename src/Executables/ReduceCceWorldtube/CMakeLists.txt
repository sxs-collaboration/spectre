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
  Boost::boost
  Boost::program_options
  Cce
  GeneralRelativity
  IO
  Informer
  Spectral
  ${SPECTRE_LIBRARIES}
  )

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES LINK_FLAGS "-nomain-module -nomain"
  )

add_dependencies(test-executables ${EXECUTABLE})
