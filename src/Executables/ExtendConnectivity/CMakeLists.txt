# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE ExtendConnectivity)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ExtendConnectivity.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  H5
  Parallel
  Spectral
  )
