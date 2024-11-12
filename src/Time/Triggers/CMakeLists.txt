# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NearTimes.cpp
  OnSubsteps.cpp
  SlabCompares.cpp
  Slabs.cpp
  StepsWithinSlab.cpp
  TimeCompares.cpp
  Times.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  NearTimes.hpp
  OnSubsteps.hpp
  SlabCompares.hpp
  Slabs.hpp
  StepsWithinSlab.hpp
  TimeCompares.hpp
  TimeTriggers.hpp
  Times.hpp
  )
