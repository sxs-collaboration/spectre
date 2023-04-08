# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PoissonSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Lorentzian.cpp
  MathFunction.cpp
  Moustache.cpp
  ProductOfSinusoids.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Lorentzian.hpp
  MathFunction.hpp
  Moustache.hpp
  ProductOfSinusoids.hpp
  Zero.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  MathFunctions
  Options
  Poisson
  Serialization
  Utilities
  )
