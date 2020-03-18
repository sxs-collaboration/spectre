# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

set(LIBRARY_SOURCES
    Averager.cpp
    Controller.cpp
    FunctionOfTimeUpdater.cpp
    TimescaleTuner.cpp
   )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  INTERFACE FunctionsOfTime
  )
