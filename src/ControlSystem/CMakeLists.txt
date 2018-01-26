# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystem)

set(LIBRARY_SOURCES
    PiecewisePolynomial.cpp
    SettleToConstant.cpp
   )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
