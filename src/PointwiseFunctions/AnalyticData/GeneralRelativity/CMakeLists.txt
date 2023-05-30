# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativityAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BrillLindquist.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  BrillLindquist.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  GeneralRelativity
  Interpolation
  Utilities
  )

if (TARGET SpEC::Exporter)
  spectre_target_sources(
    ${LIBRARY}
    PRIVATE
    SpecInitialData.cpp
    )
  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    InterpolateFromSpec.hpp
    SpecInitialData.hpp
    )
  target_link_libraries(
    ${LIBRARY}
    PUBLIC
    SpEC::Exporter
    )
endif()
