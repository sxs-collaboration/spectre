# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitializeConstraintGammas.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeConstraintGammas.hpp
  ReceiveWorldtubeData.hpp
  SendToWorldtube.hpp
)
