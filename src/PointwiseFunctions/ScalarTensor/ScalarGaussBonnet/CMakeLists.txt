# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarGaussBonnetPointwise)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CouplingParameters.cpp
  ScalarSource.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CouplingParameters.hpp
  ScalarSource.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ScalarTensor
  DataStructures
  INTERFACE
  ErrorHandling
  )
