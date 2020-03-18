# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FunctionsOfTime)

set(LIBRARY_SOURCES
  PiecewisePolynomial.cpp
  RegisterDerivedWithCharm.cpp
  SettleToConstant.cpp
  Tags.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )
