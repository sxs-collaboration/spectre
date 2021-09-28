// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the linear solver

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"

/// Functionality for solving nonlinear systems of equations
namespace NonlinearSolver {

/// Options related to nonlinear solvers
namespace OptionTags {

/*!
 * \brief Sufficient decrease parameter of the line search globalization
 *
 * The sufficient decrease parameter is the acceptable decrease of the residual
 * magnitude in each step of the nonlinear solver. It is measured as a fraction
 * of the predicted decrease in residual magnitude if the problem was linear.
 * For example, a sufficient decrease parameter of 1 means that a nonlinear
 * solver step is expected to decrease the residual exactly as expected for a
 * linear problem, i.e. immediately to zero. A sufficient decrease parameter of
 * 0.5 means that decreasing the residual by half of that amount in each
 * nonlinear solver step is acceptable.
 *
 * Nonlinear solver steps that fail the sufficient decrease condition (also
 * known as _Armijo condition_) undergo a globalization procedure such as a line
 * search.
 *
 * A typical value for the sufficient decrease parameter is \f$10^{-4}\f$. Set
 * to values closer to unity when the nonlinear solver overshoots, e.g. when the
 * initial guess is particularly bad. Larger values mean the nonlinear solver is
 * stricter with accepting steps, preferring to apply the globalization
 * strategy.
 */
template <typename OptionsGroup>
struct SufficientDecrease {
  using type = double;
  static constexpr Options::String help = {
      "Fraction of decrease predicted by linearization"};
  static type lower_bound() { return 0.; }
  static type upper_bound() { return 1.; }
  static type suggested_value() { return 1.e-4; }
  using group = OptionsGroup;
};

/*!
 * \brief Nonlinear solver steps are damped by this factor
 *
 * Instead of attempting to take full-length steps when correcting the solution
 * in each nonlinear solver step (see `NonlinearSolver::Tags::Correction`),
 * reduce the step length by this factor. This damping occurs before any
 * globalization steps that may further reduce the step length.
 */
template <typename OptionsGroup>
struct DampingFactor {
  using type = double;
  static constexpr Options::String help = {
      "Multiply corrections by this factor"};
  static type lower_bound() { return 0.; }
  static type upper_bound() { return 1.; }
  static type suggested_value() { return 1.; }
  using group = OptionsGroup;
};

/*!
 * \brief The maximum number of allowed globalization steps
 *
 * Nonlinear solves of well-posed problems should never hit this limit because
 * the step size shrinks and eventually triggers the sufficient-decrease
 * condition (see `NonlinearSolver::OptionTags::SufficientDecrease`). So the
 * suggested value just provides a safety-net to prevent the globalization
 * from running forever when the problem is ill-posed.
 */
template <typename OptionsGroup>
struct MaxGlobalizationSteps {
  using type = size_t;
  static constexpr Options::String help = {
      "Maximum number of globalization steps"};
  static type suggested_value() { return 40; }
  using group = OptionsGroup;
};

}  // namespace OptionTags

namespace Tags {

/*!
 * \brief The correction \f$\delta x\f$ to improve a solution \f$x_0\f$
 *
 * A linear problem \f$Ax=b\f$ can be equivalently formulated as the problem
 * \f$A\delta x=b-A x_0\f$ for the correction \f$\delta x\f$ to an initial guess
 * \f$x_0\f$. More importantly, we can use a correction scheme to solve a
 * nonlinear problem \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly solving a
 * linearization of it. For instance, a Newton-Raphson scheme iteratively
 * refines an initial guess \f$x_0\f$ by repeatedly solving the linearized
 * problem
 *
 * \f{equation}
 * \frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k)\delta x_k =
 * b-A_\mathrm{nonlinear}(x_k)
 * \f}
 *
 * for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 */
template <typename Tag>
struct Correction : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The nonlinear operator \f$A_\mathrm{nonlinear}\f$ applied to the data
 * in `Tag`
 */
template <typename Tag>
struct OperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    // Add "Nonlinear" prefix to abbreviate the namespace for uniqueness
    return "NonlinearOperatorAppliedTo(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The nonlinear residual
 * \f$r_\mathrm{nonlinear} = b - A_\mathrm{nonlinear}(\delta x)\f$
 */
template <typename Tag>
struct Residual : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    // Add "Nonlinear" prefix to abbreviate the namespace for uniqueness
    return "NonlinearResidual(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/// Compute the residual \f$r=b - Ax\f$ from the `SourceTag` \f$b\f$ and the
/// `db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, FieldsTag>`
/// \f$Ax\f$.
template <typename FieldsTag, typename SourceTag>
struct ResidualCompute : db::add_tag_prefix<Residual, FieldsTag>,
                         db::ComputeTag {
  using base = db::add_tag_prefix<Residual, FieldsTag>;
  using argument_tags =
      tmpl::list<SourceTag, db::add_tag_prefix<OperatorAppliedTo, FieldsTag>>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> residual,
      const typename SourceTag::type& source,
      const typename db::add_tag_prefix<OperatorAppliedTo, FieldsTag>::type&
          operator_applied_to_fields) {
    *residual = source - operator_applied_to_fields;
  }
};

/*!
 * \brief The length of nonlinear solver steps
 *
 * Instead of taking full-length nonlinear solver steps when correcting the
 * solution as detailed in `NonlinearSolver::Tags::Correction`, the correction
 * is multiplied by this step length.
 *
 * The `NonlinearSolver::Tags::DampingFactor` multiplies the initial length of
 * each step such that the nonlinear solver never takes full-length steps if the
 * damping factor is below one. The step length can be further reduced by the
 * globalization procedure. See `NonlinearSolver::NewtonRaphson` for details.
 */
template <typename OptionsGroup>
struct StepLength : db::SimpleTag {
  static std::string name() {
    return "StepLength(" + Options::name<OptionsGroup>() + ")";
  }
  using type = double;
};

/*!
 * \brief Sufficient decrease parameter of the line search globalization
 *
 * \see `NonlinearSolver::OptionTags::SufficientDecrease`
 */
template <typename OptionsGroup>
struct SufficientDecrease : db::SimpleTag {
  static std::string name() {
    return "SufficientDecrease(" + Options::name<OptionsGroup>() + ")";
  }
  using type = double;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::SufficientDecrease<OptionsGroup>>;
  static type create_from_options(const type& option) { return option; }
};

/*!
 * \brief Nonlinear solver steps are damped by this factor
 *
 * \see `NonlinearSolver::OptionTags::DampingFactor`
 */
template <typename OptionsGroup>
struct DampingFactor : db::SimpleTag {
  static std::string name() {
    return "DampingFactor(" + Options::name<OptionsGroup>() + ")";
  }
  using type = double;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::DampingFactor<OptionsGroup>>;
  static type create_from_options(const type& option) { return option; }
};

/*!
 * \brief The maximum number of allowed globalization steps
 *
 * \see `NonlinearSolver::OptionTags::MinStepLength`
 */
template <typename OptionsGroup>
struct MaxGlobalizationSteps : db::SimpleTag {
  static std::string name() {
    return "MaxGlobalizationSteps(" + Options::name<OptionsGroup>() + ")";
  }
  using type = size_t;
  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<OptionTags::MaxGlobalizationSteps<OptionsGroup>>;
  static type create_from_options(const type& option) { return option; }
};

/// Prefix indicating the `Tag` is related to the globalization procedure
template <typename Tag>
struct Globalization : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

}  // namespace Tags
}  // namespace NonlinearSolver
