# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(Ader)
add_subdirectory(BoundaryConditions)
add_subdirectory(DgSubcell)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Imex)
add_subdirectory(Initialization)
add_subdirectory(Particles)
add_subdirectory(Ringdown)
add_subdirectory(Systems)
add_subdirectory(Triggers)
add_subdirectory(VariableFixing)

set(LIBRARY "Test_Evolution")

set(LIBRARY_SOURCES
  Test_BoundaryCorrectionTags.cpp
  Test_ComputeTags.cpp
  Test_NumericInitialData.cpp
  Test_Protocols.cpp
  Test_TagsDomain.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AnalyticData
  AnalyticSolutions
  Boost::boost
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  DataBoxTestHelpers
  DgSubcell
  DomainCreators
  Domain
  Evolution
  FunctionsOfTime
  InitialDataUtilities
  LinearOperators
  Options
  Spectral
  Time
  Utilities
  )
