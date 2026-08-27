# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Exporter")

set(LIBRARY_SOURCES
  Test_Exporter.cpp
  Test_ModalTimeSeriesReader.cpp
  Test_SpacetimeInterpolator.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainCreators
  DomainStructure
  Exporter
  H5
  Informer
  LinearOperators
  Serialization
  Spectral
  Utilities
  )

add_subdirectory(Python)
