// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the linear solver

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace LinearSolver {
namespace Tags {
template <typename OptionsGroup>
struct ConvergenceCriteria;
template <typename OptionsGroup>
struct Iterations;
}  // namespace Tags
}  // namespace LinearSolver
/// \endcond

/*!
 * \ingroup LinearSolverGroup
 * \brief Functionality for solving linear systems of equations
 */
namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief The \ref DataBoxGroup tags associated with the linear solver
 */
namespace Tags {

/*!
 * \brief The operand that the local linear operator \f$A\f$ is applied to
 *
 * \details The result of the operation should be wrapped in
 * `LinearSolver::Tags::OperatorAppliedTo`.
 */
template <typename Tag>
struct Operand : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOperand(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The linear operator \f$A\f$ applied to the data in `Tag`
 */
template <typename Tag>
struct OperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOperatorAppliedTo(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Holds an `IterationId` that identifies a step in the linear solver
 * algorithm
 */
template <typename OptionsGroup>
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    return "IterationId(" + option_name<OptionsGroup>() + ")";
  }
  using type = size_t;
  template <typename Tag>
  using step_prefix = OperatorAppliedTo<Tag>;
};

/*!
 * \brief The residual \f$r=b - Ax\f$
 */
template <typename Tag>
struct Residual : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearResidual(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/// Compute the residual \f$r=b - Ax\f$ from the `SourceTag` \f$b\f$ and the
/// `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>`
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
      const db::const_item_type<SourceTag>& source,
      const db::item_type<db::add_tag_prefix<OperatorAppliedTo, FieldsTag>>&
          operator_applied_to_fields) noexcept {
    *residual = source - operator_applied_to_fields;
  }
};

template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Initial(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief The magnitude square \f$\langle \cdot,\cdot\rangle\f$ w.r.t.
 * the `LinearSolver::inner_product`
 */
template <typename Tag>
struct MagnitudeSquare : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearMagnitudeSquare(" + db::tag_name<Tag>() + ")";
  }
  using type = double;
  using tag = Tag;
};

/*!
 * \brief The magnitude \f$\sqrt{\langle \cdot,\cdot\rangle}\f$ w.r.t.
 * the `LinearSolver::inner_product`
 */
template <typename Tag>
struct Magnitude : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearMagnitude(" + db::tag_name<Tag>() + ")";
  }
  using type = double;
  using tag = Tag;
};

/*!
 * \brief Compute the `LinearSolver::Magnitude` of a tag from its
 * `LinearSolver::MagnitudeSquare`.
 */
template <typename MagnitudeSquareTag,
          Requires<tt::is_a_v<MagnitudeSquare, MagnitudeSquareTag>> = nullptr>
struct MagnitudeCompute
    : db::add_tag_prefix<Magnitude, db::remove_tag_prefix<MagnitudeSquareTag>>,
      db::ComputeTag {
  using base =
      db::add_tag_prefix<Magnitude, db::remove_tag_prefix<MagnitudeSquareTag>>;
  using return_type = double;
  static void function(const gsl::not_null<double*> magnitude,
                       const double magnitude_square) noexcept {
    *magnitude = sqrt(magnitude_square);
  }
  using argument_tags = tmpl::list<MagnitudeSquareTag>;
};

/*!
 * \brief The prefix for tags related to an orthogonalization procedurce
 */
template <typename Tag>
struct Orthogonalization : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOrthogonalization(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief A Hessenberg matrix built up during an orthogonalization procedure
 */
template <typename Tag>
struct OrthogonalizationHistory : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearOrthogonalizationHistory(" + db::tag_name<Tag>() + ")";
  }
  using type = DenseMatrix<double>;
  using tag = Tag;
};

/*!
 * \brief A set of \f$n\f$ vectors that form a basis of the \f$n\f$-th Krylov
 * subspace \f$K_n(A,b)\f$
 *
 * \details The Krylov subspace \f$K_n(A,b)\f$ spanned by this basis is the one
 * generated by the linear operator \f$A\f$ and source \f$b\f$ that are
 * represented by the tags
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * db::add_tag_prefix<LinearSolver::Tags::Operand, Tag>>` and
 * `db::add_tag_prefix<::Tags::FixedSource, Tag>`, respectively. Therefore, each
 * basis vector is of the type `db::const_item_type<db::add_tag_prefix<Operand,
 * Tag>>`.
 */
template <typename Tag>
struct KrylovSubspaceBasis : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    // No "Linear" prefix since a Krylov subspace always refers to a linear
    // operator
    return "KrylovSubspaceBasis(" + db::tag_name<Tag>() + ")";
  }
  using type = std::vector<db::const_item_type<Tag>>;
  using tag = Tag;
};

/// Indicates the `Tag` is related to preconditioning of the linear solve
template <typename Tag>
struct Preconditioned : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the linear
 * solver has converged, along with the reason for convergence.
 */
template <typename OptionsGroup>
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept {
    return "HasConverged(" + option_name<OptionsGroup>() + ")";
  }
  using type = Convergence::HasConverged;
};

/*!
 * \brief Employs the `LinearSolver::Tags::ConvergenceCriteria` to
 * determine the linear solver has converged.
 */
template <typename FieldsTag, typename OptionsGroup>
struct HasConvergedCompute : LinearSolver::Tags::HasConverged<OptionsGroup>,
                             db::ComputeTag {
 private:
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;

 public:
  using base = LinearSolver::Tags::HasConverged<OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>,
                 LinearSolver::Tags::IterationId<OptionsGroup>,
                 residual_magnitude_tag, initial_residual_magnitude_tag>;
  static void function(const gsl::not_null<return_type*> has_converged,
                       const Convergence::Criteria& convergence_criteria,
                       const size_t iteration_id,
                       const double residual_magnitude,
                       const double initial_residual_magnitude) noexcept {
    *has_converged = {convergence_criteria, iteration_id, residual_magnitude,
                      initial_residual_magnitude};
  }
};

/// Employs the `LinearSolver::Tags::Iterations` to determine the linear solver
/// has converged once it has completed a fixed number of iterations.
template <typename OptionsGroup>
struct HasConvergedByIterationsCompute
    : LinearSolver::Tags::HasConverged<OptionsGroup>,
      db::ComputeTag {
  using base = LinearSolver::Tags::HasConverged<OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<LinearSolver::Tags::Iterations<OptionsGroup>,
                 LinearSolver::Tags::IterationId<OptionsGroup>>;
  static void function(const gsl::not_null<return_type*> has_converged,
                       const size_t iterations,
                       const size_t iteration_id) noexcept {
    *has_converged = {{iterations, 0., 0.}, iteration_id, 1., 1.};
  }
};

}  // namespace Tags

/*!
 * \ingroup LinearSolverGroup
 * \brief Option tags related to the iterative linear solver
 */
namespace OptionTags {

template <typename OptionsGroup>
struct ConvergenceCriteria {
  static constexpr OptionString help =
      "Determine convergence of the linear solve";
  using type = Convergence::Criteria;
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct Iterations {
  static constexpr OptionString help = "Number of iterations to run the solver";
  using type = size_t;
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help = "Logging verbosity";
  using group = OptionsGroup;
  static type default_value() noexcept { return ::Verbosity::Quiet; }
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief `Convergence::Criteria` that determine the linear solve has converged
 *
 * \note The smallest possible residual magnitude the linear solver can reach is
 * the product between the machine epsilon and the condition number of the
 * linear operator that is being inverted. Smaller residuals are numerical
 * artifacts. Requiring an absolute or relative residual below this limit will
 * likely lead to termination by `MaxIterations`.
 *
 * \note Remember that when the linear operator \f$A\f$ corresponds to a PDE
 * discretization, decreasing the linear solver residual below the
 * discretization error will not improve the numerical solution any further.
 * I.e. the error \f$e_k=x_k-x_\mathrm{analytic}\f$ to an analytic solution
 * will be dominated by the linear solver residual at first, but even if the
 * discretization \f$Ax_k=b\f$ was exactly solved after some iteration \f$k\f$,
 * the discretization residual
 * \f$Ae_k=b-Ax_\mathrm{analytic}=r_\mathrm{discretization}\f$ would still
 * remain. Therefore, ideally choose the absolute or relative residual criteria
 * based on an estimate of the discretization residual.
 */
template <typename OptionsGroup>
struct ConvergenceCriteria : db::SimpleTag {
  static std::string name() noexcept {
    return "ConvergenceCriteria(" + option_name<OptionsGroup>() + ")";
  }
  using type = Convergence::Criteria;
  using option_tags =
      tmpl::list<LinearSolver::OptionTags::ConvergenceCriteria<OptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static Convergence::Criteria create_from_options(
      const Convergence::Criteria& convergence_criteria) noexcept {
    return convergence_criteria;
  }
};

/// A fixed number of iterations to run the linear solver
template <typename OptionsGroup>
struct Iterations : db::SimpleTag {
  static std::string name() noexcept {
    return "Iterations(" + option_name<OptionsGroup>() + ")";
  }
  using type = size_t;

  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<LinearSolver::OptionTags::Iterations<OptionsGroup>>;
  static size_t create_from_options(const size_t iterations) noexcept {
    return iterations;
  }
};

template <typename OptionsGroup>
struct Verbosity : db::SimpleTag {
  static std::string name() noexcept {
    return "Verbosity(" + option_name<OptionsGroup>() + ")";
  }
  using type = ::Verbosity;
  using option_tags =
      tmpl::list<LinearSolver::OptionTags::Verbosity<OptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static ::Verbosity create_from_options(
      const ::Verbosity& verbosity) noexcept {
    return verbosity;
  }
};
}  // namespace Tags
}  // namespace LinearSolver

namespace Tags {

template <typename OptionsGroup>
struct NextCompute<LinearSolver::Tags::IterationId<OptionsGroup>>
    : Next<LinearSolver::Tags::IterationId<OptionsGroup>>, db::ComputeTag {
  using base = Next<LinearSolver::Tags::IterationId<OptionsGroup>>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<LinearSolver::Tags::IterationId<OptionsGroup>>;
  static void function(const gsl::not_null<return_type*> next_iteration_id,
                       const size_t iteration_id) noexcept {
    *next_iteration_id = iteration_id + 1;
  }
};

}  // namespace Tags
