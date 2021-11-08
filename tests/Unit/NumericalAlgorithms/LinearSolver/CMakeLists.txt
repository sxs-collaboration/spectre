# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearSolver")

set(LIBRARY_SOURCES
  Test_BuildMatrix.cpp
  Test_ExplicitInverse.cpp
  Test_Gmres.cpp
  Test_InnerProduct.cpp
  Test_Lapack.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearSolver/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  LinearSolver
  Logging
  ParallelSchwarz
  Utilities
  )
