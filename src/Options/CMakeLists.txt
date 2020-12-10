# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Options)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Auto.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Auto.hpp
  Factory.hpp
  Options.hpp
  OptionsDetails.hpp
  ParseOptions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  Utilities
  INTERFACE
  YamlCpp
  )
