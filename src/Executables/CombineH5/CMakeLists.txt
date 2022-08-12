# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE CombineH5)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  CombineH5.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  IO
  Parallel
  Utilities
  )
