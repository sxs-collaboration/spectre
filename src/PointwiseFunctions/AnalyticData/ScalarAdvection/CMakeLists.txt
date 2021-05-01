# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarAdvectionAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Krivodonova.cpp
  Kuzmin.cpp
  Sinusoid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Krivodonova.hpp
  Kuzmin.hpp
  Sinusoid.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  PUBLIC Utilities
  )
