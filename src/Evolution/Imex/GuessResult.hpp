// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Utilities/TMPL.hpp"
#include "Evolution/Imex/Protocols/ImplicitSourceJacobian.hpp"

namespace imex {
/// Type of guess returned from an implicit sector's `initial_guess`
/// mutator.  If `ExactSolution` is returned, the implicit solve is
/// skipped.
enum class GuessResult { InitialGuess, ExactSolution };

/// Mutator for the `initial_guess` of an implicit sector that does
/// not modify the variables.  The initial guess is therefore the
/// result of the explicit step.
struct GuessExplicitResult {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  template <typename ImplicitVars>
  static std::vector<GuessResult> apply(
      const ImplicitVars& /*inhomogeneous_terms*/,
      const double /*implicit_weight*/) {
    return {};
  }
};

/// Mutator for the `jacobian` of an implicit sector that has an
/// analytic solution.  Such a sector never does numerical solves, and
/// so does not need an available jacobian.
///
/// \note The `source` mutator is always required, even if the
/// implicit equation can be solved analytically.
struct NoJacobianBecauseSolutionIsAnalytic
    : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
      tt::ConformsTo<::protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  [[noreturn]] static void apply();
};
}  // namespace imex
