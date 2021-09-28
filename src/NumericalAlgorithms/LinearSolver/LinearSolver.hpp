// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <type_traits>
#include <utility>

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/Requires.hpp"

namespace LinearSolver::Serial {

/// Registrars for linear solvers
namespace Registrars {}

/*!
 * \brief Base class for serial linear solvers that supports factory-creation.
 *
 * Derive linear solvers from this class so they can be factory-created. If your
 * linear solver supports preconditioning, derive from
 * `PreconditionedLinearSolver` instead to inherit utility that allows using any
 * other factor-creatable linear solver as preconditioner.
 */
template <typename LinearSolverRegistrars>
class LinearSolver : public PUP::able {
 protected:
  /// \cond
  LinearSolver() = default;
  LinearSolver(const LinearSolver&) = default;
  LinearSolver(LinearSolver&&) = default;
  LinearSolver& operator=(const LinearSolver&) = default;
  LinearSolver& operator=(LinearSolver&&) = default;
  /// \endcond

 public:
  ~LinearSolver() override = default;

  /// \cond
  explicit LinearSolver(CkMigrateMessage* m);
  WRAPPED_PUPable_abstract(LinearSolver);  // NOLINT
  /// \endcond

  using registrars = LinearSolverRegistrars;
  using creatable_classes = Registration::registrants<LinearSolverRegistrars>;

  virtual std::unique_ptr<LinearSolver<LinearSolverRegistrars>> get_clone()
      const = 0;

  /*!
   * \brief Solve the linear equation \f$Ax=b\f$ where \f$A\f$ is the
   * `linear_operator` and \f$b\f$ is the `source`.
   *
   * - The (approximate) solution \f$x\f$ is returned in the
   *   `initial_guess_in_solution_out` buffer, which also serves to provide an
   *   initial guess for \f$x\f$. Not all solvers take the initial guess into
   *   account, but all expect the buffer is sized correctly.
   * - The `linear_operator` must be an invocable that takes a `VarsType` as
   *   const-ref argument and returns a `SourceType` by reference. It also takes
   *   all `OperatorArgs` as const-ref arguments.
   *
   * Each solve may mutate the private state of the solver, for example to cache
   * quantities to accelerate successive solves for the same operator. Invoke
   * `reset` to discard these caches.
   */
  template <typename LinearOperator, typename VarsType, typename SourceType,
            typename... OperatorArgs, typename... Args>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> initial_guess_in_solution_out,
      const LinearOperator& linear_operator, const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args, Args&&... args) const;

  /// Discard caches from previous solves. Use before solving a different linear
  /// operator.
  virtual void reset() = 0;
};

/// \cond
template <typename LinearSolverRegistrars>
LinearSolver<LinearSolverRegistrars>::LinearSolver(CkMigrateMessage* m)
    : PUP::able(m) {}
/// \endcond

template <typename LinearSolverRegistrars>
template <typename LinearOperator, typename VarsType, typename SourceType,
          typename... OperatorArgs, typename... Args>
Convergence::HasConverged LinearSolver<LinearSolverRegistrars>::solve(
    const gsl::not_null<VarsType*> initial_guess_in_solution_out,
    const LinearOperator& linear_operator, const SourceType& source,
    const std::tuple<OperatorArgs...>& operator_args, Args&&... args) const {
  return call_with_dynamic_type<Convergence::HasConverged, creatable_classes>(
      this, [&initial_guess_in_solution_out, &linear_operator, &source,
             &operator_args, &args...](auto* const linear_solver) {
        return linear_solver->solve(initial_guess_in_solution_out,
                                    linear_operator, source, operator_args,
                                    std::forward<Args>(args)...);
      });
}

/// Indicates the linear solver uses no preconditioner. It may perform
/// compile-time optimization for this case.
struct NoPreconditioner {};

/*!
 * \brief Base class for serial linear solvers that supports factory-creation
 * and nested preconditioning.
 *
 * To enable support for preconditioning in your derived linear solver class,
 * pass any type that has a `solve` and a `reset` function as the
 * `Preconditioner` template parameter. It can also be an abstract
 * `LinearSolver` type, which means that any other linear solver can be used as
 * preconditioner. Pass `NoPreconditioner` to disable support for
 * preconditioning.
 */
template <typename Preconditioner, typename LinearSolverRegistrars>
class PreconditionedLinearSolver : public LinearSolver<LinearSolverRegistrars> {
 private:
  using Base = LinearSolver<LinearSolverRegistrars>;

 public:
  using PreconditionerType =
      tmpl::conditional_t<std::is_abstract_v<Preconditioner>,
                          std::unique_ptr<Preconditioner>, Preconditioner>;
  struct PreconditionerOption {
    static std::string name() { return "Preconditioner"; }
    // Support factory-creatable preconditioners by storing them as unique-ptrs
    using type = Options::Auto<PreconditionerType, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "An approximate linear solve in every iteration that helps the "
        "algorithm converge.";
  };

 protected:
  PreconditionedLinearSolver() = default;
  PreconditionedLinearSolver(PreconditionedLinearSolver&&) = default;
  PreconditionedLinearSolver& operator=(PreconditionedLinearSolver&&) = default;

  explicit PreconditionedLinearSolver(
      std::optional<PreconditionerType> local_preconditioner);

  PreconditionedLinearSolver(const PreconditionedLinearSolver& rhs);
  PreconditionedLinearSolver& operator=(const PreconditionedLinearSolver& rhs);

 public:
  ~PreconditionedLinearSolver() override = default;

  /// \cond
  explicit PreconditionedLinearSolver(CkMigrateMessage* m);
  /// \endcond

  void pup(PUP::er& p) override {  // NOLINT
    PUP::able::pup(p);
    if constexpr (not std::is_same_v<Preconditioner, NoPreconditioner>) {
      p | preconditioner_;
    }
  }

  /// Whether or not a preconditioner is set
  bool has_preconditioner() const {
    if constexpr (not std::is_same_v<Preconditioner, NoPreconditioner>) {
      return preconditioner_.has_value();
    } else {
      return false;
    }
  }

  /// @{
  /// Access to the preconditioner. Check `has_preconditioner()` before calling
  /// this function. Calling this function when `has_preconditioner()` returns
  /// `false` is an error.
  template <
      bool Enabled = not std::is_same_v<Preconditioner, NoPreconditioner>,
      Requires<Enabled and
               not std::is_same_v<Preconditioner, NoPreconditioner>> = nullptr>
  const Preconditioner& preconditioner() const {
    ASSERT(has_preconditioner(),
           "No preconditioner is set. Please use `has_preconditioner()` to "
           "check before trying to retrieve it.");
    if constexpr (std::is_abstract_v<Preconditioner>) {
      return **preconditioner_;
    } else {
      return *preconditioner_;
    }
  }

  template <
      bool Enabled = not std::is_same_v<Preconditioner, NoPreconditioner>,
      Requires<Enabled and
               not std::is_same_v<Preconditioner, NoPreconditioner>> = nullptr>
  Preconditioner& preconditioner() {
    ASSERT(has_preconditioner(),
           "No preconditioner is set. Please use `has_preconditioner()` to "
           "check before trying to retrieve it.");
    if constexpr (std::is_abstract_v<Preconditioner>) {
      return **preconditioner_;
    } else {
      return *preconditioner_;
    }
  }

  // Keep the function virtual so derived classes must provide an
  // implementation, but also provide an implementation below that derived
  // classes can use to reset the preconditioner
  void reset() override = 0;

 protected:
  /// Copy the preconditioner. Useful to implement `get_clone` when the
  /// preconditioner has an abstract type.
  template <
      bool Enabled = not std::is_same_v<Preconditioner, NoPreconditioner>,
      Requires<Enabled and
               not std::is_same_v<Preconditioner, NoPreconditioner>> = nullptr>
  std::optional<PreconditionerType> clone_preconditioner() const {
    if constexpr (std::is_abstract_v<Preconditioner>) {
      return has_preconditioner()
                 ? std::optional((*preconditioner_)->get_clone())
                 : std::nullopt;
    } else {
      return preconditioner_;
    }
  }

 private:
  // Only needed when preconditioning is enabled, but current C++ can't remove
  // this variable at compile-time. Keeping the variable shouldn't have any
  // noticeable overhead though.
  std::optional<PreconditionerType> preconditioner_{};
};

template <typename Preconditioner, typename LinearSolverRegistrars>
PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>::
    PreconditionedLinearSolver(
        std::optional<PreconditionerType> local_preconditioner)
    : preconditioner_(std::move(local_preconditioner)) {}

// Override copy constructors so they can clone abstract preconditioners
template <typename Preconditioner, typename LinearSolverRegistrars>
PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>::
    PreconditionedLinearSolver(const PreconditionedLinearSolver& rhs)
    : Base(rhs) {
  if constexpr (not std::is_same_v<Preconditioner, NoPreconditioner>) {
    preconditioner_ = rhs.clone_preconditioner();
  }
}
template <typename Preconditioner, typename LinearSolverRegistrars>
PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>&
PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>::operator=(
    const PreconditionedLinearSolver& rhs) {
  Base::operator=(rhs);
  if constexpr (not std::is_same_v<Preconditioner, NoPreconditioner>) {
    preconditioner_ = rhs.clone_preconditioner();
  }
  return *this;
}

/// \cond
template <typename Preconditioner, typename LinearSolverRegistrars>
PreconditionedLinearSolver<Preconditioner, LinearSolverRegistrars>::
    PreconditionedLinearSolver(CkMigrateMessage* m)
    : Base(m) {}
/// \endcond

template <typename Preconditioner, typename LinearSolverRegistrars>
void PreconditionedLinearSolver<Preconditioner,
                                LinearSolverRegistrars>::reset() {
  if constexpr (not std::is_same_v<Preconditioner, NoPreconditioner>) {
    if (has_preconditioner()) {
      preconditioner().reset();
    }
  }
}

}  // namespace LinearSolver::Serial
