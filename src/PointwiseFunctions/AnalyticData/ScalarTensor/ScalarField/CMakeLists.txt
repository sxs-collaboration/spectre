# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY STAnalyticDataScalarField)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Inverser.cpp
  SuperposedInverser.cpp
  Zero.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Factory.hpp
  Inverser.hpp
  SuperposedInverser.hpp
  Zero.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  CurvedScalarWave
  InitialDataUtilities
  Options
  Serialization
  )
