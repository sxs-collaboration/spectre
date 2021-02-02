# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_Burgers)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Outflow.cpp
  BoundaryCorrections/Test_Hll.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Characteristics.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Burgers/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Burgers
  BurgersAnalyticData
  BurgersSolutions
  DataStructures
  Time
  Utilities
  )
