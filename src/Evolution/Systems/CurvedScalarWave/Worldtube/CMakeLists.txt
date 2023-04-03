# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarWaveWorldtube)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Tags.cpp
  PunctureField.cpp
  PunctureFieldOrder0.cpp
  PunctureFieldOrder1.cpp
  PunctureFieldOrder2.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Inboxes.hpp
  SingletonChare.hpp
  Tags.hpp
  PunctureField.hpp
  Worldtube.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Domain
  Options
  Utilities
  )

add_subdirectory(SingletonActions)
add_subdirectory(ElementActions)
