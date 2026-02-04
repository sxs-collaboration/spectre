# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Exporter")

set(LIBRARY_SOURCES
  Test_Exporter.cpp
  Test_SpacetimeInterpolator.cpp
  )

if(Boost_VERSION VERSION_GREATER_EQUAL 1.81.0)
  list(APPEND LIBRARY_SOURCES Test_ModalSpacetimeInterpolator.cpp)
endif()

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainCreators
  Exporter
  H5
  Informer
  Interpolation
  Serialization
  Spectral
  Utilities
  )

add_subdirectory(Python)
