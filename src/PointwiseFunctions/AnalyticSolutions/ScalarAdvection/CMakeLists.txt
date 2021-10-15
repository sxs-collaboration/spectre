# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarAdvectionSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Krivodonova.cpp
  Kuzmin.cpp
  Sinusoid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Krivodonova.hpp
  Kuzmin.hpp
  Sinusoid.hpp
  Solutions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )
