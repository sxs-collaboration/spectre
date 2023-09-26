# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ExternalIO")

set(LIBRARY_SOURCES "")

if (TARGET FUKA::Exporter)
  list(APPEND LIBRARY_SOURCES
    Test_InterpolateFromFuka.cpp
  )
endif()

if (NOT LIBRARY_SOURCES)
  return()
endif()

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ExternalIO
  Utilities
  )
