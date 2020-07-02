# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Time
  PRIVATE
  BinaryFraction.cpp
  FullSlab.cpp
  SimpleTimes.cpp
  SplitRemaining.cpp
  )

spectre_target_headers(
  Time
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BinaryFraction.hpp
  FullSlab.hpp
  SimpleTimes.hpp
  SplitRemaining.hpp
  StepController.hpp
  )
