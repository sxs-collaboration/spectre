# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY_SOURCES
  ${LIBRARY_SOURCES}
  Actions/Test_ApplyBoundaryFluxesLocalTimeStepping.cpp
  Actions/Test_ApplyFluxes.cpp
  Actions/Test_ComputeNonconservativeBoundaryFluxes.cpp
  Actions/Test_FluxCommunication.cpp
  Actions/Test_FluxCommunicationLts.cpp
  Actions/Test_ImposeBoundaryConditions.cpp
  PARENT_SCOPE
  )
