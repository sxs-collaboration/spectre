# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Averager.cpp
  Controller.cpp
  FunctionOfTimeUpdater.cpp
  TimescaleTuner.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Averager.hpp
  Controller.hpp
  FunctionOfTimeUpdater.hpp
  TimescaleTuner.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  FunctionsOfTime
  )
