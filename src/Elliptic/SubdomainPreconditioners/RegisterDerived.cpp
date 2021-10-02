// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"

#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void register_derived_with_charm_impl() {
  Parallel::register_derived_classes_with_charm<
      LinearSolver::Serial::LinearSolver<
          tmpl::list<::LinearSolver::Serial::Registrars::Gmres<
                         ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
                             Dim, tmpl::list<Poisson::Tags::Field>>>,
                     ::LinearSolver::Serial::Registrars::ExplicitInverse>>>();
}
}  // namespace

namespace elliptic::subdomain_preconditioners {
void register_derived_with_charm() {
  register_derived_with_charm_impl<1>();
  register_derived_with_charm_impl<2>();
  register_derived_with_charm_impl<3>();
}
}  // namespace elliptic::subdomain_preconditioners
