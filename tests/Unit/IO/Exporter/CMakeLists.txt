# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Exporter")

set(LIBRARY_SOURCES
  Test_Exporter.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainCreators
  Exporter
  H5
  Informer
  Serialization
  Spectral
  Utilities
  )

add_subdirectory(Python)
