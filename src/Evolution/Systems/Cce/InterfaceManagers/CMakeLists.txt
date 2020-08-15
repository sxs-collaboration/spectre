# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Cce
  PRIVATE
  GhLocalTimeStepping.cpp
  GhLockstep.cpp
  )

spectre_target_headers(
  Cce
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GhInterfaceManager.hpp
  GhInterpolationStrategies.hpp
  GhLocalTimeStepping.hpp
  GhLockstep.hpp
  )
