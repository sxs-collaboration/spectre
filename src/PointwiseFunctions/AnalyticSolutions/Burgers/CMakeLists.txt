# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY BurgersSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Bump.cpp
  Linear.cpp
  Step.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Bump.hpp
  Linear.hpp
  Solutions.hpp
  Step.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )
