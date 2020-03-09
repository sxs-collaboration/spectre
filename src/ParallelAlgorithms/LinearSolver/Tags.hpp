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
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace LinearSolver {
namespace Tags {
struct ConvergenceCriteria;
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
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    // Add "Linear" prefix to abbreviate the namespace for uniqueness
    return "LinearIterationId";
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
  static constexpr double function(const double& magnitude_square) noexcept {
    return sqrt(magnitude_square);
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
  using type =
      std::vector<db::const_item_type<db::add_tag_prefix<Operand, Tag>>>;
  using tag = Tag;
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the linear
 * solver has converged, along with the reason for convergence.
 */
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverHasConverged"; }
  using type = Convergence::HasConverged;
};

/*
 * \brief Employs the `LinearSolver::Tags::ConvergenceCriteria` to
 * determine the linear solver has converged.
 */
template <typename FieldsTag>
struct HasConvergedCompute : LinearSolver::Tags::HasConverged, db::ComputeTag {
 private:
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>>;
  using initial_residual_magnitude_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;

 public:
  using argument_tags =
      tmpl::list<LinearSolver::Tags::ConvergenceCriteria,
                 LinearSolver::Tags::IterationId, residual_magnitude_tag,
                 initial_residual_magnitude_tag>;
  static db::const_item_type<LinearSolver::Tags::HasConverged> function(
      const Convergence::Criteria& convergence_criteria,
      const size_t& iteration_id, const double& residual_magnitude,
      const double& initial_residual_magnitude) noexcept {
    return Convergence::HasConverged(convergence_criteria, iteration_id,
                                     residual_magnitude,
                                     initial_residual_magnitude);
  }
};

}  // namespace Tags

/*!
 * \ingroup LinearSolverGroup
 * \brief Option tags related to the iterative linear solver
 */
namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags related to the iterative linear solver, e.g.
 * convergence criteria.
 */
struct Group {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct ConvergenceCriteria {
  static constexpr OptionString help =
      "Determine convergence of the linear solve";
  using type = Convergence::Criteria;
  using group = Group;
};

struct Verbosity {
  using type = ::Verbosity;
  static constexpr OptionString help = "Logging verbosity";
  using group = Group;
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
struct ConvergenceCriteria : db::SimpleTag {
  using type = Convergence::Criteria;
  using option_tags = tmpl::list<LinearSolver::OptionTags::ConvergenceCriteria>;

  static constexpr bool pass_metavariables = false;
  static Convergence::Criteria create_from_options(
      const Convergence::Criteria& convergence_criteria) noexcept {
    return convergence_criteria;
  }
};

struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;
  using option_tags = tmpl::list<LinearSolver::OptionTags::Verbosity>;

  static constexpr bool pass_metavariables = false;
  static ::Verbosity create_from_options(
      const ::Verbosity& verbosity) noexcept {
    return verbosity;
  }
};
}  // namespace Tags
}  // namespace LinearSolver

namespace Tags {

template <>
struct NextCompute<LinearSolver::Tags::IterationId>
    : Next<LinearSolver::Tags::IterationId>, db::ComputeTag {
  using argument_tags = tmpl::list<LinearSolver::Tags::IterationId>;
  static size_t function(const size_t& iteration_id) noexcept {
    return iteration_id + 1;
  }
};

}  // namespace Tags
