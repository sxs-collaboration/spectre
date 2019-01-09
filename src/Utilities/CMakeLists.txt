# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Utilities)

set(LIBRARY_SOURCES
  FileSystem.cpp
  Formaline.cpp
  PrettyType.cpp
  Rational.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
