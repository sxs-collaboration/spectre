# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Utilities)

set(LIBRARY_SOURCES
  FileSystem.cpp
  Formaline.cpp
  OptimizerHacks.cpp
  PrettyType.cpp
  Rational.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blaze
  Boost::boost
  Brigand
  ErrorHandling
  Libxsmm
  )

if (NOT USE_FORMALINE)
  set_source_files_properties(
    Formaline.cpp
    PROPERTIES
    COMPILE_DEFINITIONS SPECTRE_NO_FORMALINE)
endif (NOT USE_FORMALINE)
