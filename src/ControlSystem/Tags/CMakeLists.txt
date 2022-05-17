# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PUBLIC
  FunctionsOfTimeInitialize.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FunctionsOfTimeInitialize.hpp
  MeasurementTimescales.hpp
  )
