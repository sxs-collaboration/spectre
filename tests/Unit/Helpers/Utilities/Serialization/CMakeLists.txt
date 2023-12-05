# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SerializationHelpers")

set(LIBRARY_SOURCES
  Versioning.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Framework
  Serialization
  Utilities
  PRIVATE
  Informer
  )
