# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE ExportEquationOfStateForRotNS)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ExportEquationOfStateForRotNS.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::boost
  Boost::program_options
  DataStructures
  Hydro
  Parallel
  Utilities
  )
