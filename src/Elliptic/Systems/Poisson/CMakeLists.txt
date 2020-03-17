# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Poisson)

set(LIBRARY_SOURCES
  Equations.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativity
  LinearOperators
  )
