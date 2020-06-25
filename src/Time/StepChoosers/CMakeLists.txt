# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Time
  PRIVATE
  Constant.cpp
  )

spectre_target_headers(
  Time
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ByBlock.hpp
  Cfl.hpp
  Constant.hpp
  Increase.hpp
  PreventRapidIncrease.hpp
  StepChooser.hpp
  StepToTimes.hpp
  )
