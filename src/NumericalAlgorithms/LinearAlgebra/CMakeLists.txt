# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearAlgebra)

set(LIBRARY_SOURCES
  FindGeneralizedEigenvalues.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
