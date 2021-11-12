# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Averager.cpp
  Controller.cpp
  TimescaleTuner.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Averager.hpp
  Component.hpp
  Controller.hpp
  Event.hpp
  Metafunctions.hpp
  NamespaceDocs.hpp
  RunCallbacks.hpp
  Tags.hpp
  TimescaleTuner.hpp
  Trigger.hpp
  UpdateFunctionOfTime.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  FunctionsOfTime
  )

add_subdirectory(Actions)
add_subdirectory(Protocols)
add_subdirectory(Tags)
