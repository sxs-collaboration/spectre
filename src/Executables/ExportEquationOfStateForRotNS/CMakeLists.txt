# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DumpRotNSEos)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DumpRotNSEos.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DumpRotNSEos.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Hydro
  Utilities
  PRIVATE
  Interpolation
  Printf
  )

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
  DumpRotNSEos
  Options
  Printf
  Utilities
  )
