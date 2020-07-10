# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AlfvenWave.cpp
  BondiMichel.cpp
  KomissarovShock.cpp
  SmoothFlow.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlfvenWave.hpp
  BondiMichel.hpp
  KomissarovShock.hpp
  SmoothFlow.hpp
  Solutions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  Hydro
  Options
  )
