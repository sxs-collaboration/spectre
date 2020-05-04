# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Informer
  IO
  LinearOperators
  MathFunctions
  Options
  ParallelLinearSolver
  Poisson
  Utilities
  )

function(add_poisson_executable
  EXECUTABLE_NAME POISSON_SYSTEM INITIAL_GUESS BOUNDARY_CONDITIONS)
  add_spectre_parallel_executable(
    "SolvePoisson${EXECUTABLE_NAME}"
    SolvePoissonProblem
    Elliptic/Executables/Poisson
    "Metavariables<${POISSON_SYSTEM}, ${INITIAL_GUESS}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_poisson_executable)

# Poisson system with Lorentzian solution
# The solution is only implemented in 3D so far
add_poisson_executable(
  Lorentzian3D
  "Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>"
  Poisson::Solutions::Lorentzian<3>
  Poisson::Solutions::Lorentzian<3>
  )
target_link_libraries(
  SolvePoissonLorentzian3D
  PRIVATE
  PoissonSolutions
  )

# Poisson system with Moustache solution
# This is a 1D or 2D solution
function(add_poisson_moustache_executable DIM)
  add_poisson_executable(
    Moustache${DIM}D
    "Poisson::FirstOrderSystem<${DIM}, Poisson::Geometry::Euclidean>"
    Poisson::Solutions::Moustache<${DIM}>
    Poisson::Solutions::Moustache<${DIM}>
    "PoissonSolutions"
    )
  target_link_libraries(
    SolvePoissonMoustache${DIM}D
    PRIVATE
    PoissonSolutions
    )
endfunction(add_poisson_moustache_executable)
add_poisson_moustache_executable(1)
add_poisson_moustache_executable(2)

# Poisson system with ProductOfSinusoids solution
function(add_poisson_product_of_sinusoids_executable DIM)
  add_poisson_executable(
    ProductOfSinusoids${DIM}D
    "Poisson::FirstOrderSystem<${DIM}, Poisson::Geometry::Euclidean>"
    Poisson::Solutions::ProductOfSinusoids<${DIM}>
    Poisson::Solutions::ProductOfSinusoids<${DIM}>
    "PoissonSolutions"
    )
  target_link_libraries(
    SolvePoissonProductOfSinusoids${DIM}D
    PRIVATE
    PoissonSolutions
    )
endfunction(add_poisson_product_of_sinusoids_executable)
add_poisson_product_of_sinusoids_executable(1)
add_poisson_product_of_sinusoids_executable(2)
add_poisson_product_of_sinusoids_executable(3)
