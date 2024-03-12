# Distributed under the MIT License.
# See LICENSE.txt for details.

# We can't do the instantiations for NewtonianEuler in the NewtonianEuler
# library because the TimeDerivativeTerms and system depend on the analytic
# solution/data being used, which causes a cyclic dependency since the solutions
# depend on the system for the tags. This cyclic dependency is already present
# in the code, but currently only through header files. The same cyclic
# dependency exists between NewtonianEulerSources and NewtonianEuler. To resolve
# these, we will need to factor the NewtonianEuler tags into a separate library.
# fter the NewtonianEuler tags are factored in this way, the instantiation
# source file should be moved to the NewtonianEuler library.
set(LIBRARY NewtonianEulerInstantiations)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  VolumeTermsInstantiation.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Evolution
  FiniteDifference
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerSolutions
  NewtonianEulerSources
  Utilities
  )

set(LIBS_TO_LINK
  Actions
  Charmxx::main
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  EventsAndDenseTriggers
  EventsAndTriggers
  Evolution
  FiniteDifference
  Hydro
  Informer
  Limiters
  LinearOperators
  MathFunctions
  NewtonianEuler
  NewtonianEulerAnalyticData
  NewtonianEulerInstantiations
  NewtonianEulerLimiters
  NewtonianEulerSolutions
  NewtonianEulerSources
  Observer
  Options
  Parallel
  PhaseControl
  Serialization
  Time
  Utilities
  )

function(add_newtonian_euler_executable DIM)
  set(EXECUTABLE "EvolveNewtonianEuler${DIM}D")
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    EvolveNewtonianEuler.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_newtonian_euler_executable(1)
add_newtonian_euler_executable(2)
add_newtonian_euler_executable(3)
