# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Constant.cpp
  LimitIncrease.cpp
  Maximum.cpp
  PreventRapidIncrease.cpp
  Random.cpp
  StepToTimes.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ByBlock.hpp
  Cfl.hpp
  Constant.hpp
  ElementSizeCfl.hpp
  ErrorControl.hpp
  Factory.hpp
  LimitIncrease.hpp
  Maximum.hpp
  PreventRapidIncrease.hpp
  Random.hpp
  StepChooser.hpp
  StepToTimes.hpp
  )
