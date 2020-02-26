# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Integration)

set(LIBRARY_SOURCES
  GslQuadAdaptive.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  GSL::gsl
  Utilities
  )

