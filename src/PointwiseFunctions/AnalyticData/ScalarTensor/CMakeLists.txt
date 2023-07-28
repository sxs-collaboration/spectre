# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarTensorAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  KerrSphericalHarmonic.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  KerrSphericalHarmonic.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  CurvedScalarWave
  InitialDataUtilities
  Options
  Serialization
  )
