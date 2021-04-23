# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarAdvection)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  System.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
