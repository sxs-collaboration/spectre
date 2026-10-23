# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "IoTestHelpers")

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  H5Identifiers.cpp
  VolumeData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  H5Identifiers.hpp
  VolumeData.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  H5
  PRIVATE
  Boost::boost
  DataStructures
  ErrorHandling
  Framework
  Utilities
  )

add_subdirectory(Observers)
