# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_RayTracerBackgroundSpacetimes")

set(LIBRARY_SOURCES
  Test_NumericData.cpp
  Test_WrappedGr.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Domain
  DomainCreators
  DomainStructure
  Framework
  GeneralRelativity
  GeneralRelativitySolutions
  H5
  RayTracer
  Spectral
  Utilities
  )
