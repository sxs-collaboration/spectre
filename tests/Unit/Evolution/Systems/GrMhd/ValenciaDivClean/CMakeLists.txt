# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ValenciaDivClean")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Subcell/Test_InitialDataTci.cpp
  Subcell/Test_SwapGrTags.cpp
  Subcell/Test_TciOnDgGrid.cpp
  Subcell/Test_TciOnFdGrid.cpp
  Subcell/Test_TciOptions.cpp
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_FixConservatives.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  Test_ValenciaDivClean.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd/ValenciaDivClean"
  "${LIBRARY_SOURCES}"
  "DomainHelpers;Framework;GeneralRelativityHelpers;HydroHelpers;ValenciaDivClean"
  )
