# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsEvents")

set(LIBRARY_SOURCES
  Test_ObserveAdmIntegrals.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainStructure
  Elliptic
  GeneralRelativitySolutions
  LinearOperators
  SpecialRelativity
  XctsEvents
  XctsSolutions
  )
