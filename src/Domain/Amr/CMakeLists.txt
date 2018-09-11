# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Amr)

set(LIBRARY_SOURCES
  Flag.cpp
  Helpers.cpp
  UpdateAmrDecision.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE Domain
  INTERFACE ErrorHandling
  )
