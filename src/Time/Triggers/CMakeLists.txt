# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NearTimes.cpp
  SlabCompares.cpp
  Slabs.cpp
  TimeCompares.cpp
  Times.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  NearTimes.hpp
  SlabCompares.hpp
  Slabs.hpp
  TimeCompares.hpp
  TimeTriggers.hpp
  Times.hpp
  )
