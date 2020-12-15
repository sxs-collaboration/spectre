# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PoissonSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Lorentzian.cpp
  Moustache.cpp
  ProductOfSinusoids.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  Lorentzian.hpp
  Moustache.hpp
  ProductOfSinusoids.hpp
  Solutions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  )
