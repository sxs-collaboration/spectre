# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Elliptic
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgElementArray.hpp
  ImposeBoundaryConditions.hpp
  ImposeInhomogeneousBoundaryConditionsOnSource.hpp
  InitializeFirstOrderOperator.hpp
  )

add_subdirectory(NumericalFluxes)
