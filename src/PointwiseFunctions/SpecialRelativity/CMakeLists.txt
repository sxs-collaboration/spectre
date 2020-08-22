# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SpecialRelativity)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LorentzBoostMatrix.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LorentzBoostMatrix.hpp
  )

target_link_libraries(
  ${LIBRARY}
  )
