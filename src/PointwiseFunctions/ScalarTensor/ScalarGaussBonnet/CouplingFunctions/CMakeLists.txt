# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarGaussBonnetCouplings)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CouplingFunction.cpp
  Exponential.cpp
  QuarticPolynomial.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CouplingFunction.hpp
  Exponential.hpp
  Factory.hpp
  QuarticPolynomial.hpp
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
