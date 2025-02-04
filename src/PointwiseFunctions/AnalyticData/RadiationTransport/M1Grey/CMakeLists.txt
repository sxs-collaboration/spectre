# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY M1GreyAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  HomogeneousSphere.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Factory.hpp
  HomogeneousSphere.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  Hydro
  Options
  )
