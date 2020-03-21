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
  Utilities
  ${SPECTRE_LIBRARIES}
  )

function(add_linear_elliptic_executable
  EXECUTABLE_NAME ELLIPTIC_SYSTEM INITIAL_GUESS BOUNDARY_CONDITIONS EXTRA_LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "Solve${EXECUTABLE_NAME}"
    SolveLinearEllipticProblem
    Elliptic/Executables/Linear
    "Metavariables<${ELLIPTIC_SYSTEM}, ${INITIAL_GUESS}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK};${EXTRA_LIBS_TO_LINK}"
    )
endfunction(add_linear_elliptic_executable)

# Poisson system with Lorentzian solution
# The solution is only implemented in 3D so far
add_linear_elliptic_executable(
  PoissonLorentzian3D
  "Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>"
  Poisson::Solutions::Lorentzian<3>
  Poisson::Solutions::Lorentzian<3>
  "Poisson;PoissonSolutions"
  )

# Poisson system with Moustache solution
# This is a 1D or 2D solution
function(add_poisson_moustache_executable DIM)
  add_linear_elliptic_executable(
    PoissonMoustache${DIM}D
    "Poisson::FirstOrderSystem<${DIM}, Poisson::Geometry::Euclidean>"
    Poisson::Solutions::Moustache<${DIM}>
    Poisson::Solutions::Moustache<${DIM}>
    "Poisson;PoissonSolutions"
    )
endfunction(add_poisson_moustache_executable)
add_poisson_moustache_executable(1)
add_poisson_moustache_executable(2)

# Poisson system with ProductOfSinusoids solution
function(add_poisson_product_of_sinusoids_executable DIM)
  add_linear_elliptic_executable(
    PoissonProductOfSinusoids${DIM}D
    "Poisson::FirstOrderSystem<${DIM}, Poisson::Geometry::Euclidean>"
    Poisson::Solutions::ProductOfSinusoids<${DIM}>
    Poisson::Solutions::ProductOfSinusoids<${DIM}>
    "Poisson;PoissonSolutions"
    )
endfunction(add_poisson_product_of_sinusoids_executable)
add_poisson_product_of_sinusoids_executable(1)
add_poisson_product_of_sinusoids_executable(2)
add_poisson_product_of_sinusoids_executable(3)
