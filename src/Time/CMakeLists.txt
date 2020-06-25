# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Time)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Slab.cpp
  Time.cpp
  TimeStepId.cpp
  Utilities.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryHistory.hpp
  EvolutionOrdering.hpp
  History.hpp
  Slab.hpp
  Tags.hpp
  Time.hpp
  TimeStepId.hpp
  Utilities.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  ErrorHandling
  Options
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(StepChoosers)
add_subdirectory(StepControllers)
add_subdirectory(TimeSteppers)
add_subdirectory(Triggers)
