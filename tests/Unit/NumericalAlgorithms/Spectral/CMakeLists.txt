# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory("Python")

set(LIBRARY "Test_Spectral")

set(LIBRARY_SOURCES
  Test_ChebyshevGauss.cpp
  Test_ChebyshevGaussLobatto.cpp
  Test_Clenshaw.cpp
  Test_Filtering.cpp
  Test_FiniteDifference.cpp
  Test_IndefiniteIntegral.cpp
  Test_LegendreGauss.cpp
  Test_LegendreGaussLobatto.cpp
  Test_LogicalCoordinates.cpp
  Test_Mesh.cpp
  Test_Projection.cpp
  Test_Spectral.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  ErrorHandling
  LinearOperators
  Spectral
  Utilities
  )
