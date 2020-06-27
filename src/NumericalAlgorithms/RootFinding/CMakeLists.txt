# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RootFinding)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GslMultiRoot.cpp
  QuadraticEquation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GslMultiRoot.hpp
  NewtonRaphson.hpp
  QuadraticEquation.hpp
  TOMS748.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  ErrorHandling
  GSL::gsl
  )
