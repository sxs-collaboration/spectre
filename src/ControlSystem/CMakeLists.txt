# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

set(LIBRARY_SOURCES
    Controller.cpp
    PiecewisePolynomial.cpp
    SettleToConstant.cpp
    TimescaleTuner.cpp
   )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
