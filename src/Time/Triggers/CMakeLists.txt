# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Time
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  NearTimes.hpp
  PastTime.hpp
  Slabs.hpp
  SpecifiedTimes.hpp
  TimeTriggers.hpp
  )
