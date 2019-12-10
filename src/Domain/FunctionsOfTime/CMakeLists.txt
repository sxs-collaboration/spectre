# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FunctionsOfTime)

set(LIBRARY_SOURCES
  PiecewisePolynomial.cpp
  SettleToConstant.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
