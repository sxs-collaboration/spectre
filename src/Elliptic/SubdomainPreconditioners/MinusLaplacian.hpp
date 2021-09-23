// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// Linear solvers that approximately invert the
/// `elliptic::dg::subdomain_operator::SubdomainOperator` to make the Schwarz
/// subdomain solver converge faster.
/// \see LinearSolver::Schwarz::Schwarz
namespace elliptic::subdomain_preconditioners {

/// \cond
template <size_t Dim, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
struct MinusLaplacian;
/// \endcond

namespace Registrars {
template <size_t Dim, typename OptionsGroup,
          typename Solver = LinearSolver::Serial::LinearSolver<tmpl::list<
              ::LinearSolver::Serial::Registrars::Gmres<
                  ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
                      Dim, tmpl::list<Poisson::Tags::Field>>>,
              ::LinearSolver::Serial::Registrars::ExplicitInverse>>>
struct MinusLaplacian {
  template <typename LinearSolverRegistrars>
  using f = subdomain_preconditioners::MinusLaplacian<Dim, OptionsGroup, Solver,
                                                      LinearSolverRegistrars>;
};
}  // namespace Registrars

/*!
 * \brief Approximate the subdomain operator with a flat-space Laplacian with
 * Dirichlet boundary conditions for every tensor component separately.
 *
 * This linear solver applies the `Solver` to every tensor component in
 * turn, approximating the subdomain operator with a flat-space Laplacian with
 * Dirichlet boundary conditions. This can be a lot cheaper than solving the
 * full subdomain operator and can provide effective preconditioning for an
 * iterative subdomain solver. The approximation is better the closer the
 * original PDEs are to a set of decoupled flat-space Poisson equations with
 * Dirichlet boundary conditions.
 *
 * \tparam Dim Spatial dimension
 * \tparam OptionsGroup The options group identifying the
 * `LinearSolver::Schwarz::Schwarz` solver that defines the subdomain geometry.
 * \tparam Solver Any class that provides a `solve` and a `reset` function,
 * but typically a `LinearSolver::Serial::LinearSolver`. The solver will be
 * factory-created from input-file options.
 */
template <size_t Dim, typename OptionsGroup,
          typename Solver = LinearSolver::Serial::LinearSolver<tmpl::list<
              ::LinearSolver::Serial::Registrars::Gmres<
                  ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
                      Dim, tmpl::list<Poisson::Tags::Field>>>,
              ::LinearSolver::Serial::Registrars::ExplicitInverse>>,
          typename LinearSolverRegistrars =
              tmpl::list<Registrars::MinusLaplacian<Dim, OptionsGroup, Solver>>>
class MinusLaplacian
    : public LinearSolver::Serial::LinearSolver<LinearSolverRegistrars> {
 private:
  using Base = LinearSolver::Serial::LinearSolver<LinearSolverRegistrars>;
  using StoredSolverType = tmpl::conditional_t<std::is_abstract_v<Solver>,
                                               std::unique_ptr<Solver>, Solver>;

 public:
  static constexpr size_t volume_dim = Dim;
  using options_group = OptionsGroup;
  using poisson_system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
  using SubdomainOperator =
      elliptic::dg::subdomain_operator::SubdomainOperator<poisson_system,
                                                          OptionsGroup>;
  using SubdomainData = ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, tmpl::list<Poisson::Tags::Field>>;
  using solver_type = Solver;

  struct SolverOptionTag {
    static std::string name() { return "Solver"; }
    using type = StoredSolverType;
    static constexpr Options::String help =
        "The linear solver used to invert the Laplace operator. The solver is "
        "shared between the tensor components.";
  };

  using options = tmpl::list<SolverOptionTag>;
  static constexpr Options::String help =
      "Approximate the linear operator with a Laplace operator with Dirichlet "
      "boundary conditions for every tensor component separately.";

  MinusLaplacian() = default;
  MinusLaplacian(MinusLaplacian&& /*rhs*/) = default;
  MinusLaplacian& operator=(MinusLaplacian&& /*rhs*/) = default;
  ~MinusLaplacian() = default;
  MinusLaplacian(const MinusLaplacian& rhs)
      : Base(rhs), solver_(rhs.clone_solver()) {}
  MinusLaplacian& operator=(const MinusLaplacian& rhs) {
    Base::operator=(rhs);
    solver_ = rhs.clone_solver();
    return *this;
  }

  /// \cond
  explicit MinusLaplacian(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MinusLaplacian);  // NOLINT
  /// \endcond

  MinusLaplacian(StoredSolverType solver) : solver_(std::move(solver)) {}

  const Solver& solver() const {
    if constexpr (std::is_abstract_v<Solver>) {
      return *solver_;
    } else {
      return solver_;
    }
  }

  /// Solve the equation \f$Ax=b\f$ by approximating \f$A\f$ with a Laplace
  /// operator with homogeneous Dirichlet boundary conditions for every tensor
  /// component in \f$x\f$.
  template <typename LinearOperator, typename VarsType, typename SourceType,
            typename... OperatorArgs>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> solution, LinearOperator&& linear_operator,
      const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args = std::tuple{}) const;

  void reset() override { mutable_solver().reset(); }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Base::pup(p);
    p | solver_;
  }

  std::unique_ptr<Base> get_clone() const override {
    return std::make_unique<MinusLaplacian>(*this);
  }

 private:
  Solver& mutable_solver() {
    if constexpr (std::is_abstract_v<Solver>) {
      return *solver_;
    } else {
      return solver_;
    }
  }

  StoredSolverType clone_solver() const {
    if constexpr (std::is_abstract_v<Solver>) {
      return solver_->get_clone();
    } else {
      return solver_;
    }
  }

  StoredSolverType solver_{};

  // These are memory buffers that are kept around for repeated solves. Possible
  // optimization: Free this memory at appropriate times, e.g. when the element
  // has completed a full subdomain solve and goes to the background. In some
  // cases the `subdomain_operator_` is never even used again in subsequent
  // subdomain solves because it is cached as a matrix (see
  // LinearSolver::Serial::ExplicitInverse), so we don't need the memory anymore
  // at all.
  mutable SubdomainOperator subdomain_operator_{
      // Use Dirichlet boundary conditions. Note that the _linearized_ boundary
      // conditions are applied by the subdomain operator, so the constant is
      // irrelevant. Possible optimization: Choose a boundary condition for
      // every external boundary in the input file, so they fit better to the
      // real boundary conditions. In particular, the correct choice Dirichlet
      // vs. Neumann b.c. is relevant for the effectiveness of the
      // preconditioner.
      std::make_unique<Poisson::BoundaryConditions::Robin<
          Dim, typename poisson_system::boundary_conditions_base::registrars>>(
          1., 0., 0.)};
  mutable SubdomainData source_{};
  mutable SubdomainData initial_guess_in_solution_out_{};
};

namespace detail {
template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
void assign_component(
    const gsl::not_null<::LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, LhsTagsList>*>
        lhs,
    const ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, RhsTagsList>& rhs,
    const size_t lhs_component, const size_t rhs_component) {
  // Possible optimization: Once we have non-owning Variables we can use a view
  // into the rhs here instead of copying.
  const size_t num_points_element = rhs.element_data.number_of_grid_points();
  for (size_t i = 0; i < num_points_element; ++i) {
    lhs->element_data.data()[lhs_component * num_points_element + i] =
        rhs.element_data.data()[rhs_component * num_points_element + i];
  }
  for (const auto& [overlap_id, rhs_data] : rhs.overlap_data) {
    const size_t num_points_overlap = rhs_data.number_of_grid_points();
    // The random-access operation is relatively slow because it computes a
    // hash, so it's important for performance to avoid repeating it in every
    // iteration of the loop below.
    auto& lhs_vars = lhs->overlap_data[overlap_id];
    for (size_t i = 0; i < num_points_overlap; ++i) {
      lhs_vars.data()[lhs_component * num_points_overlap + i] =
          rhs_data.data()[rhs_component * num_points_overlap + i];
    }
  }
}
}  // namespace detail

template <size_t Dim, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
template <typename LinearOperator, typename VarsType, typename SourceType,
          typename... OperatorArgs>
Convergence::HasConverged
MinusLaplacian<Dim, OptionsGroup, Solver, LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*> initial_guess_in_solution_out,
    LinearOperator&& /*linear_operator*/, const SourceType& source,
    const std::tuple<OperatorArgs...>& operator_args) const {
  source_.destructive_resize(source);
  initial_guess_in_solution_out_.destructive_resize(source);
  // Solve each component of the source variables in turn, assuming the operator
  // is a Laplacian
  for (size_t component = 0;
       component < source.element_data.number_of_independent_components;
       ++component) {
    detail::assign_component(make_not_null(&source_), source, 0, component);
    detail::assign_component(make_not_null(&initial_guess_in_solution_out_),
                             *initial_guess_in_solution_out, 0, component);
    solver().solve(make_not_null(&initial_guess_in_solution_out_),
                   subdomain_operator_, source_, operator_args);
    detail::assign_component(initial_guess_in_solution_out,
                             initial_guess_in_solution_out_, component, 0);
  }
  return {0, 0};
}

/// \cond
template <size_t Dim, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID MinusLaplacian<Dim, OptionsGroup, Solver,
                                 LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace elliptic::subdomain_preconditioners
