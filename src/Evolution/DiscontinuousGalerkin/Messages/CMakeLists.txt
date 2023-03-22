# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(BoundaryMessage)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryMessage.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryMessage.cpp
  )

add_dependencies(
  ${LIBRARY}
  module_BoundaryMessage
  )
