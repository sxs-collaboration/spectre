# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Evolution)

set(LIBRARY_SOURCES
  TagsDomain.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Domain
  DataStructures
  Utilities

  PRIVATE
  LinearOperators
  )

add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Executables)
add_subdirectory(Systems)
add_subdirectory(VariableFixing)
