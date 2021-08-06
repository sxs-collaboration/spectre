# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GhLocalTimeStepping.cpp
  GhLockstep.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GhInterfaceManager.hpp
  GhLocalTimeStepping.hpp
  GhLockstep.hpp
  )
